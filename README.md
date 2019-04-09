# ZSY2
Training an Deep Q agent to learn ZSY.

争上游 (ZhengShangYou, or “Competition Upstream”) is a Chinese card game that is part strategy, part luck. Each player is dealt about 18 cards, they get rid of cards by matching patterns, and the player who gets rid of all their cards first wins.

Play the 2 player version of the game now! Tutorial level included for those new to ZSY. Game build with Unity.

[Windows Build](https://drive.google.com/file/d/1auOngmTju8Pg8WQKNkGNUvjCXRWxkJOA/view?fbclid=IwAR3N98b22L4ZyDtSRqkbiMfywEdkoRwJEjCaFVm6kDdDDWaqjxwsHM1zHSE)

[Mac Build](https://drive.google.com/file/d/1ZQ_-JJ__go3fGu9P5BJSzQ3dYoFxi0rY/view?fbclid=IwAR1iPtRYhbjz0xf1JY3TXDvfldyzaf2au4NtP9svqFuBKH9LiLaoWxiUADY)

In 2018, I attempted to create a Deep Q Agent to play this game better than humans, which you can read about [here](https://github.com/leonl0000/ZSY/blob/master/writeups/CS%20230%20Project%20final%20writeup.pdf) and [here](https://github.com/leonl0000/ZSY/blob/master/writeups/poster.png). To make a long story short, it failed by only having a win rate of ~40%. The agent was too simple, the training did not incorporate more advanced RL methods like fixed targets, and I didn't even have a good sample of humans to test it against.

This time, I seem to have reached human level performance. The 3 key improvements were: Conv nets, assembly models, and battle royale exploration. Read the final writeup [here](Writeups/Final Report.pdf)


