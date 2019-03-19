using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class tutorialBoxScript : MonoBehaviour
{
    public static readonly string[] texts = {
        "  ZSY is a card game that is part strategy, part luck.\n  Each player is dealt 18 cards.\n  The first to get rid of them wins the game",
        "  This is your hand.\n  You can select cards by clicking on them.",
        "  This is the table, where the cards currently in play will be.\n  Once you've selected cards from your hand, you can play them by clicking on the table.",
        "  This is the opponents hand.\n  You can see how many cards they have, but not what they are",
        "  Play a 3 by clicking on the 3 and clicking on the table,",
        //4
        "  After you play a card, your opponent has to either play a higher card or pass.\n  In this case they played a 7.",
        "  You could beat that with an 8, but for now just play a 9.",
        //6
        "  Ah! An Ace!\n  The only way you can beat that is with your Joker",
        //7
        "  Since your opponent couldn't beat your Joker, they passed.\n  You win this round, and get to start the next round.",
        "  The player who starts a round gets to decide what pattern is played.\n  Last round, the pattern was 'singles'.",
        "  You can also play 'doubles' or 'triples'.\n  For example, you have a triple 6.",
        "  You can select many cards at once by clicking and dragging your mouse over the cards.\n  Cancel a selection by clicking in the blue area.",
        "  Select the triple 6 and play it by clicking on the table",
        //12
        "  Since you set the pattern to be 'triples', your opponent can only match with a higher 'triple'.\n  In this case, triple 9s.",
        "  But look!\n  You have triple 10s you can play.",
        //14
        "  They can't beat that, so they passed.\n\n  All that's left to explain are <b>chains</b> and <b>bombs</b>.",
        "  A <b>chain</b> is any set of consecutive cards where there's each 'link' in the chain a double or a triple\n (or quadruple, but we'll get to that later).",
        "  Right now, you have a <b>chain</b> that you can play: [77 88]",
        //17
        "  To beat a <b>chain</b>, you have to have cards that are higher and match the pattern exactly.\n",
        "  In this case, they beat your\n[77 88] with [QQ KK]",
        "  For now, pass by clicking on the table without selecting any cards",
        //20
        "  What they just played is also a <b>chain</b>: 3, 4, and 5 are in a row, and are either doubles <i>or</i> triples",
        "  To beat it with a <b>chain</b>, you would need that exact pattern\n<double triple double>, but with higher cards",
        "  For example, if you had [55 666 77], that would work\n\n  [55 666 88] would not because 5, 6, and 8 are not consecutive",
        "  [666 77 88] would also not work: 6, 7, and 8 are consecutive, but it's a <triple double double> not a \n<double triple double>",
        "  The longer the <b>chain</b>, the harder it is to counter. But! There is another way.",
        "  Any 4 of a kind is a <b>bomb</b>\n  A bomb can counter any single, double, triple or chain at any time.\n  ",
        "  The only counter to a bomb is a bigger bomb.\n  So, play your bomb!",
        //27
        "  Looks like they can't beat your bomb, so it's your turn again. Play your last card and win the game.",
        //28
        "<align=\"center\">You win!</align>",
        "  Now that you're done with the tutorial, play against one of the actual AIs:",
        "  <b>DQN</b> is a AI agent trained with the hope of matching human level performance. Read more about it in the About page.",
        "  If you're not quite up for facing that, maybe warm up by playing facing one of the other two AIs:",
        "  <b>Greedy</b> is a rather dumb AI that always tries to get rid of as many cards as possible",
        "  <b>Random</b> is an even dumber AI that takes moves completely at random.",
        "  One last thing: You can change the pace of the game in the settings button in the top left of the screen",
        "  Good luck, and thanks for playing!",
        //36
    };

    public static readonly float[][] pos = {
        new float[2] {0, 0},
        new float[2] {1, -1},
        new float[2] {1, 0},
        new float[2] {1, 1},
        new float[2] {0, 0}, //4

        new float[2] {1, 0},
        new float[2] {1, -1}, //6

        new float[2] {-1, 0}, //7

        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {1, -1}, //11

        new float[2] {1, 0},
        new float[2] {1, 0}, //13

        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {1, -1}, //16

        new float[2] {-1, 0},
        new float[2] {-1, 0},
        new float[2] {-1, 0}, //19

        new float[2] {1, 0},
        new float[2] {1, 0},
        new float[2] {1, 0},
        new float[2] {1, 0},
        new float[2] {1, 0},
        new float[2] {1, 0},
        new float[2] {1, -1}, //26

        new float[2] {1, -1}, //27

        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {0, 0},
        new float[2] {0, 0} //35
    };

    public void Set(int index) {
        //Debug.Log("tbs " + index);
        Vector3 r = .3f * Screen.width * Vector3.right;
        Vector3 u = .3f * Screen.height * Vector3.up;
        transform.Find("Text").GetComponent<TextMeshProUGUI>().text = texts[index];
        transform.position = transform.position + r * pos[index][0] + u * pos[index][1];
        transform.Find("Button").GetComponent<Button>().onClick.AddListener(delegate {
            GameObject.Find("Canvas").GetComponent<zsyGame>().TutorialFunction();
        });
    }
    
}
