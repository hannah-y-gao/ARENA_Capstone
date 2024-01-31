# ARENA_Capstone
Capstone project for ARENA 3.0

**Observations from plots of top 3 commonly predicted tokens** 
*Top 3 commonly predicted tokens for every head in GPT2-small*
![Top 3 commonly predicted tokens for every head in GPT2-small](imgs/MT.png, "Monday --> Tuesday prediction")
![Top 3 commonly predicted tokens for every head in GPT2-small](imgs/TW.png, "Tueday --> Wednesday prediction")
![Top 3 commonly predicted tokens for every head in GPT2-small](imgs/WTh.png, "Wednesday --> Thursday prediction")
![Top 3 commonly predicted tokens for every head in GPT2-small](imgs/ThF.png, "Thursday --> Friday prediction")
![Top 3 commonly predicted tokens for every head in GPT2-small](imgs/FS.png, "Friday --> Saturday prediction")
![Top 3 commonly predicted tokens for every head in GPT2-small](imgs/SSu.png, "Saturday --> Sunday prediction")

- Head (10, 3) consistently predicts days in its top three; however, it almost always most strongly predicts the *subject day* rather than the correct *subsequent day*
    - The top 3 days predicted by this head always includes the correct *subsequent day*
    - Exception: this head predicts the correct *subsequent day* when the *subject day* is Tuesday (i.e. it most strongly predicts Wednesday)
- Head (8, 1) also consistently predicts days in its top three (somewhat at random); it strongly predicts the correct token following Tueday
- 