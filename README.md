# Recommendation_sys

## Overview
Using a russian online marketplace company Avito's dataset containing features like:
1. UserID
2. ProductID
3. AdID
4. UserDemoraphics
5. AdPosition
6. Click
.... and many more.

The main aim was to produce personalized product recommendation based on the user interaction with products and similarity with other users. I have tried to prevent any monotonus recommendations
by keep the accuracy low to include different products. LightFM model was used to include both user and Ad meta deta to determine the same.
 
Apart from that, a classification model is also implemented just to personlize the location too. This model is not completely tuned and required more data analysis. 
