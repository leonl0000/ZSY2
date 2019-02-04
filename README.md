# ZSY2
Training an Deep Q agent to learn ZSY.


争上游 (ZhengShangYou, or “Competition Upstream”) is a Chinese card game that is part strategy, part luck. Each player is dealt about 18 cards, they get rid of cards by matching patterns, and the player who gets rid of all their cards first wins.

In 2018, I attempted to create a Deep Q Agent to play this game better than humans, which you can read about [here](https://github.com/leonl0000/ZSY/blob/master/writeups/CS%20230%20Project%20final%20writeup.pdf) and [here](https://github.com/leonl0000/ZSY/blob/master/writeups/poster.png). To make a long story short, it failed by only having a win rate of ~30%. The agent was too simple, the training did not incorporate more advanced RL methods like fixed targets, and I didn't even have a good sample of humans to test it against.

I'm doing it again, properly this time.
