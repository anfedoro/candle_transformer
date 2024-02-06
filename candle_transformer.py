
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import model_train as mt
from dataset  import prepare_data, unscale_data

import time

#caclulate number of trainable parameters
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {params} trainable parameters')

def hl_str(text):
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    return f'{RED}{text}{RESET}'


def train(model, optimizer, scaler, loss_fn, train_loader, val_loader, writer, device, hp, data_scaler):

    #unpack hyperparameters
    start_epoch = hp['start_epoch']
    num_epochs = hp['num_epochs']
    embed_dim = hp['embed_dim']
    seq_len = hp['seq_len']
    best_vloss = hp['best_vloss']
    best_loss = hp['best_loss']
    best_mape = hp['best_mape']
 
    data_mean = data_scaler['mean'][:,:4]
    data_std = data_scaler['std'][:,:4]
    
    warmup_steps = hp['warmup_steps']
    l_rates, max_lr = mt.get_leaning_rates(embed_dim, warmup_steps, num_epochs)

    max_train_batches = train_loader.num_batches
    max_val_batches = val_loader.num_batches
    

    start_time = time.perf_counter()
    #set model to train mode
    try:
        for epoch in range(start_epoch, num_epochs):
            # Train the model
            wd = 0 if epoch <= warmup_steps else 1e-7
            optimizer = mt.set_learning_rate(optimizer, l_rates[epoch], weight_decay=wd)
            if epoch == 0:
                print(f'Model training with initial learning rate: {next(iter(optimizer.param_groups))["lr"]:.2e} and '
                      f'weight decay: {next(iter(optimizer.param_groups))["weight_decay"]:.2e} for {num_epochs} epochs.\n'
                      f'Max.learning rate: {max_lr:.2e} at epoch {warmup_steps}')
            model.train()
            epoch_loss = 0
            toc = time.perf_counter()
            data_len = 0
            samples_num = None
            for batch_idx, data, labels in train_loader:
               
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    # Forward pass
                    outputs = model(data)


                    loss = loss_fn(outputs, labels)

                epoch_loss += loss.item()

                # Backward and optimize
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                

                tic = time.perf_counter() - toc
                data_len += data.shape[0]
                step_time = data_len / tic
                if samples_num is None:
                    samples_num = max_train_batches * data.shape[0] 
                    
                estimated_time = samples_num /step_time    
                time_since_start = int(time.perf_counter() - start_time)
                formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'
                str_ = f'Training time: {formatted_time}, Epoch: [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{max_train_batches}], Loss: {epoch_loss / (batch_idx + 1):.8f}, Instant loss:{loss.item():.8f}, Iter time: {step_time:.0f} smpls/sec Est.time: {estimated_time:.2f} sec     '
        
                print(str_, end='\r', flush=True)
                
                # if batch_idx >=5:
                #     break


            #add weights and biases to tensorboard
            if epoch % 1 == 0:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        writer.add_histogram(f'weights/{name}', param, epoch)
                    elif 'bias' in name:
                        writer.add_histogram(f'biases/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'grads/{name}', param.grad, epoch)
            
                    

            print(' ' * len(str_), end='\r', flush=True)

            # Test the model
            
            model.eval()
            vepoch_loss = 0
            mape = 0
            with torch.inference_mode():
                
                for vbatch_idx, vdata, vlabels in val_loader:
                    
                    vdata = vdata.to(device)
                    vlabels = vlabels.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                        # Forward pass
                        voutputs = model(vdata)
                        
                        vloss = loss_fn(voutputs, vlabels)
                        mape += mt.mape_loss(voutputs, vlabels).item()
                        
                    vepoch_loss += vloss.item()

           
                    
                    time_since_start = int(time.perf_counter() - start_time)
                    formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'

                    print(f'Training time: {formatted_time}, Epoch [{epoch + 1}/{num_epochs}], '
                          f'Step [{vbatch_idx + 1}/{max_val_batches}], Validation Loss: {vepoch_loss/(vbatch_idx +1):.8f}, '
                          f'MAPE error: {mape/(vbatch_idx +1):.2f}%'
                          , end='\r', flush=True)

                    

            time_since_start = int(time.perf_counter() - start_time)
            formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'
            if epoch_loss / max_train_batches < best_loss:
                best_loss = epoch_loss / max_train_batches
                best_loss_str = hl_str(f'{best_loss:.8f}')
            else:
                best_loss_str = f'{epoch_loss / max_train_batches:.8f}'
            
            if vepoch_loss / max_val_batches < best_vloss:
                best_vloss_str = hl_str(f'{vepoch_loss / max_val_batches:.8f}')
            else:
                best_vloss_str = f'{vepoch_loss / max_val_batches:.8f}'
            
            if mape / max_val_batches < best_mape:
                best_mape = mape / max_val_batches
                best_mape_str = hl_str(f'{best_mape:.2f}')
            else:
                best_mape_str = f'{mape / max_val_batches:.2f}'
            

            lr = next(iter(optimizer.param_groups))['lr']
            log_str = f'Training time: {formatted_time}, Epoch [{epoch + 1}/{num_epochs}], ' + \
                  f'Train Loss: {best_loss_str}, Validation Loss: {best_vloss_str}, MAPE error: {best_mape_str}%, Learning rate: {lr:.2e} '
            
            print(log_str, end='\n', flush=True)
            

            writer.add_scalars('Loss', {'Train': epoch_loss / max_train_batches, 'Test': vepoch_loss / max_val_batches}, epoch)
            writer.add_scalar('MAPE error', mape / max_val_batches, epoch)

            #unscale date to restore the original values
            vlabels_unscaled = unscale_data(vlabels[:60,-1].cpu().detach(), data_mean, data_std)
            voutputs_unscaled = unscale_data(voutputs[:60,-1].cpu().detach().to(torch.float32), data_mean, data_std)

            mt.write_charts_to_TB('Labels', writer, vlabels_unscaled.numpy(), epoch)
            mt.write_charts_to_TB('Predictions', writer,voutputs_unscaled.numpy() , epoch)
        

            
            if vepoch_loss/max_val_batches < best_vloss:
                best_vloss = vepoch_loss/max_val_batches
                hp['best_vloss'] = best_vloss
                hp['best_loss'] = best_loss
                hp['best_mape'] = best_mape

                mt.store_traning_result(model, optimizer, scaler, epoch, best_model_path, hp)

                
                time_since_start = int(time.perf_counter() - start_time)
                formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'
                log_str = f'Training time: {formatted_time}, The best model saved at epoch {epoch+1} with validation loss {best_vloss_str}'
                print(log_str)

                
            
            writer.flush()
        is_exeption = False

    except KeyboardInterrupt:
        print(f"Interrupted at epoch {epoch}")
        is_exeption = True

    except Exception as e:
        print(f"Exception at epoch {epoch}: {e}")
        is_exeption = True

    finally:
        time_since_start = int(time.perf_counter() - start_time)
        formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'
        if not is_exeption:
            epoch += 1
        hp['best_vloss'] = best_vloss
        hp['best_loss'] = best_loss
        hp['best_mape'] = best_mape

        mt.store_traning_result(model, optimizer, scaler, epoch, last_model_path, hp)
        writer.close()
        print(f'Training time: {formatted_time}, The last model saved at epoch {epoch+1} with validation loss {best_vloss:.8f}')
        print('Training completed.')

def main():
    
    #load data
    batch_size = 1024
    seq_len = 61
    

    continue_training = True
    is_best = False
    print('Prepare data...', end='', flush=True)
    train_loader, val_loader, data_scaler = prepare_data('data/USATECHIDXUSD_1s.csv', seq_len, batch_size, shuffle=True)
    print('Done.')


    global best_model_path
    global last_model_path
    global log_dir
    best_model_path = './models/candle_transformer_best_2.pth'
    last_model_path = './models/candle_transformer_last_2.pth'
    log_dir = 'runs/candle_transformer_2'


        #Unpack hyperparameters and initialize model

    #initialize hyperparameters
    hp = {'start_epoch':0, 
            'num_epochs':1000,
            'warmup_steps':100,
            'input_dim':6,
            'output_dim':4, 
            'seq_len':seq_len,
            'embed_dim':128, 
            'num_heads':8, 
            'num_layers':8, 
            'dropout':0.05, 
            'ff_mult':4,
            'best_vloss':float('inf'),
            'best_loss':float('inf'),
            'best_mape': float('inf')
           }

    #initialize model, loss function and optimizer
    model, device = mt.init_model(hp)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    optimizer = optim.AdamW(model.parameters(), weight_decay=0)
   
    if continue_training:
        #load model from checkpoint
        load_path = best_model_path if is_best else last_model_path

        print(f'Loading model from checkpoint {load_path}')
      
        model, optimizer, scaler, epoch, hyperparameters = mt.load_model(model, optimizer, scaler, load_path, device)

        train_loss_str = hl_str(f'{hyperparameters["best_loss"]:.8f}')
        test_loss_str = hl_str(f'{hyperparameters["best_vloss"]:.8f}')
        mape_str = hl_str(f'{hyperparameters["best_mape"]:.2f}%')
        
        # epoch = 103
        print(f'The best model loaded from checkpoint. Epoch: {epoch+1}, Train Loss: {train_loss_str}, Validation Loss: {test_loss_str}, MAPE error: {mape_str}')

        #update hyperparameters
        hp['start_epoch'] = epoch + 1 if is_best else epoch
        hp['best_vloss'] = hyperparameters['best_vloss']
        hp['best_loss'] = hyperparameters['best_loss']
        hp['best_mape'] = hyperparameters['best_mape']

        purge_step = epoch if is_best else None
      
    else:
        purge_step = None
        writers = None

    print(f'Model hyperparameters are:\nInput_size: {hp["input_dim"]},\tSeq_len: {hp["seq_len"]},\tEmbed_dim: {hp["embed_dim"]}\nNum_heads: {hp["num_heads"]},\tNum_layers: {hp["num_layers"]},\tDropout: {hp["dropout"]}, FF_mult: {hp["ff_mult"]}')
    print(f'Optimizer to be used is {type(optimizer)}')
    print(f'Batch size: {batch_size}, train batches: {train_loader.num_batches}, test batches: {val_loader.num_batches}')
    count_parameters(model)

    
    writer = mt.init_writer(log_dir, continue_training=continue_training, purge_step=purge_step, writers=None)


    
    #train model
    loss_fn = mt.SeqWeightedMSELoss(weight=5.0)
    train(model, optimizer, scaler, loss_fn, train_loader, val_loader, writer, device, hp, data_scaler)
    
if __name__ == '__main__':
    main()
