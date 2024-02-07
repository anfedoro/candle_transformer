This is another yeat attempt to create apply transformer model for candlestick chart forecasting.

In few words I am creating set of candle features based on log of their price returns (ratios of open price to past candle clos and rations of HLC to the open). This approach allow to keep time sequence stationar to be able to perform autoregression.
So in general the approcch and the model itself is relatively simple. The only difference of general transformer is that I don use any Embedding on the input words.. as I obviously have no words (tokens) but continious data instead. I am expanding input features to higher dimentiona space using fc laier and then apply Embedding layer to the sequence to perform positional encoding. Al the rest if pretty much generic transformer.

Frankly, the result is not yet promising (perhaps I simply have no anough patience to wait until the traning completed :)). I am observing that the model restore input sequence of candles (minus the first one) pretty easy, while prediction of new last candle still an issue. I am continue to work on this and will update the repository in case of any improvements.
