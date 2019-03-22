using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;
using UnityEngine.UI;
using TensorFlow;
using TMPro;
using UnityEngine.EventSystems;

public class zsyGame : MonoBehaviour
{
    public TFGraph graph;
    private TFSession session;

    public static List<Tuple<string, int>> CardAndVals = new List<Tuple<string, int>>() {
        new Tuple<string, int> ("3C", 0), new Tuple<string, int> ("3D", 0), new Tuple<string, int> ("3H", 0), new Tuple<string, int> ("3S", 0),
        new Tuple<string, int> ("4C", 1), new Tuple<string, int> ("4D", 1), new Tuple<string, int> ("4H", 1), new Tuple<string, int> ("4S", 1),
        new Tuple<string, int> ("5C", 2), new Tuple<string, int> ("5D", 2), new Tuple<string, int> ("5H", 2), new Tuple<string, int> ("5S", 2),
        new Tuple<string, int> ("6C", 3), new Tuple<string, int> ("6D", 3), new Tuple<string, int> ("6H", 3), new Tuple<string, int> ("6S", 3),
        new Tuple<string, int> ("7C", 4), new Tuple<string, int> ("7D", 4), new Tuple<string, int> ("7H", 4), new Tuple<string, int> ("7S", 4),
        new Tuple<string, int> ("8C", 5), new Tuple<string, int> ("8D", 5), new Tuple<string, int> ("8H", 5), new Tuple<string, int> ("8S", 5),
        new Tuple<string, int> ("9C", 6), new Tuple<string, int> ("9D", 6), new Tuple<string, int> ("9H", 6), new Tuple<string, int> ("9S", 6),
        new Tuple<string, int> ("10C", 7), new Tuple<string, int> ("10D", 7), new Tuple<string, int> ("10H", 7), new Tuple<string, int> ("10S", 7),
        new Tuple<string, int> ("JC", 8), new Tuple<string, int> ("JD", 8), new Tuple<string, int> ("JH", 8), new Tuple<string, int> ("JS", 8),
        new Tuple<string, int> ("QC", 9), new Tuple<string, int> ("QD", 9), new Tuple<string, int> ("QH", 9), new Tuple<string, int> ("QS", 9),
        new Tuple<string, int> ("KC", 10), new Tuple<string, int> ("KD", 10), new Tuple<string, int> ("KH", 10), new Tuple<string, int> ("KS", 10),
        new Tuple<string, int> ("AC", 11), new Tuple<string, int> ("AD", 11), new Tuple<string, int> ("AH", 11), new Tuple<string, int> ("AS", 11),
        new Tuple<string, int> ("2C", 12), new Tuple<string, int> ("2D", 12), new Tuple<string, int> ("2H", 12), new Tuple<string, int> ("2S", 12),
        new Tuple<string, int> ("BJK", 13), new Tuple<string, int> ("RJK", 14)};

    public static int[] inds = new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53};

    public static int[] emptyMove = new int[15] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    public static void PrintArr(int[] arr) { Debug.Log(string.Join(", ", arr)); }
    public static readonly bool[] allIncreasing = new bool[15] {
        true, true, true, true, true, true, true, true, true, true, true, true, true, true, true };

    public int[] PlayerHand;
    public int[] AgentHand;
    public List<int[]> History;

    List<List<CardButton>> cbs;
    List<List<CardButton>> Agent_cbs;
    public int[] curSelected;
    public bool[] isIncreasing = new bool[15];
    public int[] curMove;
    public List<int[]> LegalMoves;

    public int playerScore;
    public int computerScore;


    public RandomAgent randomAgent;
    public GreedyAgent greedyAgent;
    public LearnedAgent learnedAgent;
    public Agent PlayingAgent;
    public float AgentDelayTimeout;
    public float AgentDelayTimer;
    public bool AgentIsDelaying;

    public void SetPlayerButtonStatus(bool b) {
        for(int i=0; i<15; i++) {
            foreach (CardButton cb in cbs[i])
                cb.bt.enabled = b;
        }
    }

    public void DelayAgentPlay() {
        SetPlayerButtonStatus(false);
        AgentDelayTimer = AgentDelayTimeout;
        AgentIsDelaying = true;
    }

    public void clearTable() {
        foreach (Transform child in TableTransform)
            if(child.tag != "dontClear") Destroy(child.gameObject);
    }
    public void clearAll() {
        foreach (Transform child in AgentHandTransform)
            if (child.name.Contains("Card")) Destroy(child.gameObject);
        clearTable();
        foreach (Transform child in PlayerHandTransform)
            if (child.name.Contains("Card")) Destroy(child.gameObject);
    }
    public void clearSelection() {
        foreach (List<CardButton> _cbs in cbs)
            foreach (CardButton cb in _cbs)
                cb.SetUnselected();
        curSelected = new int[15];
        allIncreasing.CopyTo(isIncreasing, 0);
    }
    public void setScoreText() {ScoreText.text = string.Format("Human: {0} \t\t\t\t\t\t\tComputer: {1}", playerScore, computerScore);}

    public void CardClick(int cardVal, bool isSelected) {
        if (isSelected) curSelected[cardVal] += 1;
        else curSelected[cardVal] -= 1;
        bool MoveIsLegal = false;
        foreach (int[] move in LegalMoves) MoveIsLegal |= curSelected.SequenceEqual(move);
    }

    public void DebugButtonClick() {
        PrintArr(curMove);
        PrintArr(curSelected);
        PrintArr(PlayerHand);
    }

    public void PlayCards() {
        bool MoveIsLegal = false;
        foreach (int[] move in LegalMoves) MoveIsLegal |= curSelected.SequenceEqual(move);
        if(!MoveIsLegal) {
            manager.Popup("illegal move", Color.red);
            return;
        }
        clearTable();
        float w = 41;
        float f = Screen.width / 631f;
        curMove = new int[15];
        curSelected.CopyTo(curMove, 0);
        History.Add(curMove);
        for (int i=0; i<15; i++) {
            PlayerHand[i] -= curSelected[i];
            for (int j = cbs[i].Count-1; j > -1; j--) {
                if (curSelected[i] == 0) break;
                if (cbs[i][j].getSelected()) {
                    curSelected[i] -= 1;
                    cbs[i][j].bg.transform.SetParent(TableTransform, false);
                    cbs[i][j].bg.transform.position += 12 * (curSelected[i] - j) * Vector3.up * f;
                    cbs[i][j].SetUnselected();
                    cbs[i][j].bt.enabled = false;
                    cbs[i].RemoveAt(j);
                    for(int k=j; k<cbs[i].Count; k++)
                        cbs[i][k].bg.transform.position -= 12 * Vector3.up * f;
                }
            }
        }
        TableTransform.GetComponent<Image>().color = PlayerHandTransform.GetComponent<Image>().color;
        curSelected = new int[15];
        allIncreasing.CopyTo(isIncreasing, 0);
        if (PlayerHand.SequenceEqual(emptyMove)) {
            playerScore += 1;
            ScoreText.text = string.Format("Human:{0} \t\t\t\t\t\t\tComputer:{1}", playerScore, computerScore);
            if (TutorialCounter == 0) manager.postGame(true);
            else TutorialFunction();
        } else DelayAgentPlay();
    }

    public void AgentTakeTurn() {
        List<int[]> agentLegalMoves = GetLegalMoves(AgentHand, curMove);
        Tuple<int[], float> moveAndConf = PlayingAgent.GetMove(AgentHand, agentLegalMoves, History);    
        clearTable();
        int[] move = moveAndConf.Item1;
        float f = Screen.width / 631f;
        for (int i = 0; i < 15; i++) {
            if (move[i] == 0) continue;
            for (int j = move[i] - 1; j > -1; j--) {
                Agent_cbs[i][j].bg.transform.position -= 300 * Vector3.up * f;
                Agent_cbs[i][j].bg.transform.SetParent(TableTransform, false);
                Agent_cbs[i].RemoveAt(j);
            }
            for (int j = 0; j < Agent_cbs[i].Count; j++)
                Agent_cbs[i][j].bg.transform.position -= 12 * move[i] * Vector3.up * f;
            AgentHand[i] -= move[i];
        }
        TableTransform.GetComponent<Image>().color = AgentHandTransform.GetComponent<Image>().color;
        curMove = move;
        History.Add(curMove);
        AgentCardCountText.text = "" + AgentHand.Sum();
        ConfidenceText.text = showConfidence ? string.Format("Confidence {0:00.00}%", PlayingAgent.confidence * 100) : "";
        curSelected = new int[15];
        if(AgentHand.SequenceEqual(emptyMove)) {
            computerScore += 1;
            ScoreText.text = string.Format("Human: {0} \t\t\t\t\t\t\tComputer: {1}", playerScore, computerScore);
            manager.postGame(false);
        } else {
            LegalMoves = GetLegalMoves(PlayerHand, curMove);
            SetPlayerButtonStatus(true);
            if (TutorialCounter != 0) TutorialFunction();
        }
    }

    public void StartGame(Agent agent = null) {
        clearAll();
        int[] shuffledInds = inds.OrderBy(x => Random.value).ToArray();
        List<Tuple<string, int>> PlayerCards = new List<Tuple<string, int>>();
        List<Tuple<string, int>> AgentCards = new List<Tuple<string, int>>();
        PlayerHand = new int[15];
        int[] PlayerHandTemp = new int[15];
        int[] AgentHandTemp = new int[15];
        AgentHand = new int[15];
        for (int i = 0; i < 18; i++) {
            PlayerCards.Add(CardAndVals[shuffledInds[i]]);
            PlayerHand[CardAndVals[shuffledInds[i]].Item2]++;
            PlayerHandTemp[CardAndVals[shuffledInds[i]].Item2]++;
            AgentCards.Add(CardAndVals[shuffledInds[i + 18]]);
            AgentHand[CardAndVals[shuffledInds[i + 18]].Item2]++;
            AgentHandTemp[CardAndVals[shuffledInds[i + 18]].Item2]++;
        }
        PlayerCards = PlayerCards.OrderBy(o => -o.Item2).ToList();
        AgentCards = AgentCards.OrderBy(o => -o.Item2).ToList();
        float w = 41;
        float f = Screen.width / 631f;
        cbs = new List<List<CardButton>>();
        Agent_cbs = new List<List<CardButton>>();
        for (int i = 0; i < 15; i++) {
            cbs.Add(new List<CardButton>());
            Agent_cbs.Add(new List<CardButton>());
        }
        foreach (Tuple<string, int> cav in PlayerCards) {
            CardButton cb = CardButton.NewCardButton(cav.Item1, CardClick, cav.Item2);
            cb.bg.transform.SetParent(PlayerHandTransform, false);
            cb.bg.transform.position = new Vector3((started ? 30: -285) + cav.Item2 * w, 30 + PlayerHandTemp[cav.Item2] * 12, 0) * f;
            cbs[cav.Item2].Insert(0, cb);
            PlayerHandTemp[cav.Item2] -= 1;
        }
        foreach (Tuple<string, int> cav in AgentCards) {
            CardButton cb = CardButton.NewCardButton(cav.Item1);
            cb.bg.transform.SetParent(AgentHandTransform, false);
            cb.bg.transform.position = new Vector3((started ? 30 : -285) + cav.Item2 * w, (started? 565: 330) + AgentHandTemp[cav.Item2] * 12, 0) * f;
            cb.bt.enabled = false;
            Agent_cbs[cav.Item2].Insert(0, cb);
            AgentHandTemp[cav.Item2] -= 1;
        }

        started = true;
        AgentCardCountText.text = "18";
        ConfidenceText.text = "";

        //Init a bunch of zeros stuffs
        History = new List<int[]>();
        curSelected = new int[15];
        allIncreasing.CopyTo(isIncreasing, 0);
        curMove = new int[15];
        int turn = Random.Range(0, 2);
        if (agent != null) PlayingAgent = agent;
        if (turn == 1) DelayAgentPlay();
        else LegalMoves = GetOpeningMoves(PlayerHand);
    }

    public void StartTutorialGame() {
        clearAll();
        AgentDelayTimeout = 1.2f;
        int[] shuffledInds = new int[] {
            0, 11, 12, 15, 13, 16, 17, 22, 23, 25, 29, 31, 28, 32, 33, 34, 35, 52,
            2, 3, 7, 4, 5, 10, 9, 14, 19, 20, 26, 27, 24, 37, 36, 42, 40, 44,
        };
        List<Tuple<string, int>> PlayerCards = new List<Tuple<string, int>>();
        List<Tuple<string, int>> AgentCards = new List<Tuple<string, int>>();
        PlayerHand = new int[15];
        int[] PlayerHandTemp = new int[15];
        int[] AgentHandTemp = new int[15];
        AgentHand = new int[15];
        for (int i = 0; i < 18; i++) {
            PlayerCards.Add(CardAndVals[shuffledInds[i]]);
            PlayerHand[CardAndVals[shuffledInds[i]].Item2]++;
            PlayerHandTemp[CardAndVals[shuffledInds[i]].Item2]++;
            AgentCards.Add(CardAndVals[shuffledInds[i + 18]]);
            AgentHand[CardAndVals[shuffledInds[i + 18]].Item2]++;
            AgentHandTemp[CardAndVals[shuffledInds[i + 18]].Item2]++;
        }
        PlayerCards = PlayerCards.OrderBy(o => -o.Item2).ToList();
        AgentCards = AgentCards.OrderBy(o => -o.Item2).ToList();

        float w = 41;
        float f = Screen.width / 631f;
        cbs = new List<List<CardButton>>();
        Agent_cbs = new List<List<CardButton>>();
        for (int i = 0; i < 15; i++) {
            cbs.Add(new List<CardButton>());
            Agent_cbs.Add(new List<CardButton>());
        }
        foreach (Tuple<string, int> cav in PlayerCards) {
            CardButton cb = CardButton.NewCardButton(cav.Item1, CardClick, cav.Item2);
            cb.bg.transform.SetParent(PlayerHandTransform, false);
            cb.bg.transform.position = new Vector3((started ? 30 : -285) + cav.Item2 * w, 30 + PlayerHandTemp[cav.Item2] * 12, 0) * f;
            cbs[cav.Item2].Insert(0, cb);
            PlayerHandTemp[cav.Item2] -= 1;
        }
        foreach (Tuple<string, int> cav in AgentCards) {
            CardButton cb = CardButton.NewCardButton(cav.Item1);
            cb.bg.transform.SetParent(AgentHandTransform, false);
            cb.bg.transform.position = new Vector3((started ? 30 : -285) + cav.Item2 * w, (started ? 565 : 330) + AgentHandTemp[cav.Item2] * 12, 0) * f;
            cb.bt.enabled = false;
            Agent_cbs[cav.Item2].Insert(0, cb);
            AgentHandTemp[cav.Item2] -= 1;
        }

        started = true;
        AgentCardCountText.text = "18";
        ConfidenceText.text = "";

        //Init a bunch of zeros stuffs
        History = new List<int[]>();
        curSelected = new int[15];
        allIncreasing.CopyTo(isIncreasing, 0);
        curMove = new int[15];
        PlayingAgent = new TutorialAgent();
        LegalMoves = GetOpeningMoves(PlayerHand);
        PlayCardsButton.transform.SetAsFirstSibling();
        tutorialObject = Resources.Load<GameObject>("TutorialObject");
        cv2 = GameObject.Find("Canvas2");
        TutorialFunction();
    }
    public void callTrueTutorial() {
        TutorialFunction(true);
    }
    private GameObject cv2;
    public int TutorialCounter = 0;
    private GameObject tutorialObject;
    GameObject previousTutorial;
    private void TutorialBox(int i) {
        Destroy(previousTutorial);
        GameObject g = Instantiate(tutorialObject);
        tutorialBoxScript tbs = g.GetComponent<tutorialBoxScript>();
        g.transform.SetParent(cv2.transform, false);
        tbs.Set(i);
        previousTutorial = g;
        TutorialCounter++;
    }
    private void TutorialPlayCards(int[] cardsToPlay, int nextCounter) {
        if (curSelected.SequenceEqual(cardsToPlay)) {
            TutorialCounter = nextCounter;
            Destroy(previousTutorial);
            PlayCards();
        } else {
            if (TutorialCounter == nextCounter-1) manager.Popup("Please follow the tutorial", Color.green);
            else TutorialCounter++;
        }
    }
    public void quitTutorial() {
        Destroy(previousTutorial);
    }
    public void TutorialFunction(bool isTable = false) {
        switch(TutorialCounter) {
            case int n when n < 5:
                if (!isTable) TutorialBox(TutorialCounter);
                break;

            case 5:
            case 6:
                TutorialPlayCards(new int[15] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 7);
                break;


            case 7:
            case 8:
                TutorialBox(TutorialCounter - 2);
                break;

            case 9:
            case 10:
                TutorialPlayCards(new int[15] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, 11);
                break;

            case 11:
                TutorialBox(7);
                break;

            case 12:
            case 13:
                TutorialPlayCards(new int[15] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, 14);
                break;

            case int n when (n >= 14 && n < 19):
                TutorialBox(TutorialCounter - 6);
                break;

            case 19:
            case 20:
                TutorialPlayCards(new int[15] { 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 21);
                break;

            case 21:
            case 22:
                TutorialBox(TutorialCounter - 8);
                break;

            case 23:
            case 24:
                TutorialPlayCards(new int[15] { 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0 }, 25);
                break;

            case int n when (n >= 25 && n < 28):
                TutorialBox(TutorialCounter - 10);
                break;

            case 28:
            case 29:
                TutorialPlayCards(new int[15] { 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 30);
                break;

            case int n when (n >= 30 && n < 33):
                TutorialBox(TutorialCounter - 12);
                break;

            case 33:
            case 34:
                TutorialPlayCards(new int[15] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 35);
                break;

            case int n when (n >= 35 && n < 42):
                TutorialBox(TutorialCounter - 14);
                break;

            case 42:
            case 43:
                TutorialPlayCards(new int[15] { 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0 }, 44);
                break;

            case 44:
                TutorialBox(28);
                break;

            case 45:
            case 46:
                TutorialPlayCards(new int[15] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 47);
                break;

            case int n when (n >= 46 && n < 54):
                TutorialBox(TutorialCounter - 18);
                break;

            default:
                TutorialCounter = 0;
                PlayCardsButton.transform.SetAsLastSibling();
                AgentDelayTimeout = 0.5f;
                manager.mainMenu();
                Destroy(previousTutorial);
                break;
        }
    }

    public List<int[]> GetOpeningMoves(int[] hand) {
        var runner = session.GetRunner();
        runner.AddInput(graph["hand"][0], hand);
        runner.Fetch(graph["ret"][0]);
        int[,] output = runner.Run()[0].GetValue() as int[,];
        List<int[]> moves = new List<int[]>();
        for (int i = 0; i < output.GetLength(0); i++)
            moves.Add(new int[15] {output[i, 0], output[i, 1], output[i, 2], output[i, 3], output[i, 4],
                output[i, 5], output[i, 6], output[i, 7], output[i, 8], output[i, 9],
                output[i, 10], output[i, 11], output[i, 12], output[i, 13], output[i, 14]});
        return moves;
    }
    public List<int[]> GetLegalMoves(int[] hand, int[] PrevMove) {
        if(PrevMove.SequenceEqual(emptyMove)) return GetOpeningMoves(hand);
        List<int[]> moves = new List<int[]>() { emptyMove };
        int lowCard = -1;
        int highCard = -1;
        for(int i=0; i<15; i++) {
            if(PrevMove[i] != 0) {
                if (lowCard == -1) lowCard = i;
                highCard = i;
            } 
        }
        for(int i=lowCard+1; i<15-highCard+lowCard; i++) {
            bool isLegal = true;
            int[] newMove = new int[15];
            for (int j = 0; j <= highCard - lowCard; j++) {
                isLegal &= hand[i + j] >= PrevMove[lowCard + j];
                newMove[i + j] = PrevMove[lowCard + j];
            }
            if (isLegal) moves.Add(newMove);            
        }
        if ((lowCard == highCard) && PrevMove[lowCard] == 4) return moves;
        //If the move is a bomb, return moves, else check for legal bombs
        for(int i=0; i<13; i++) {
            if(hand[i] == 4) {
                int[] newMove = new int[15];
                newMove[i] = 4;
                moves.Add(newMove);
            }
        }
        return moves;
    }


    public Manager manager;
    public Transform PlayerHandTransform;
    public Transform TableTransform;
    public Button PlayCardsButton;
    public Transform AgentHandTransform;
    public TextMeshProUGUI ScoreText;
    public TextMeshProUGUI AgentCardCountText;
    public TextMeshProUGUI ConfidenceText;
    public bool showConfidence;
    private bool started = false;
    void Start()
    {
        started = true;
        graph = new TFGraph();
        TextAsset graphModel2 = Resources.Load<TextAsset>("Opt_OpM");
        graph.Import(graphModel2.bytes);
        session = new TFSession(graph);

        manager = GameObject.Find("Manager").GetComponent<Manager>();

        PlayerHandTransform = transform.Find("PlayerHand");
        TableTransform = transform.Find("Table");
        PlayCardsButton = TableTransform.Find("PlayCardsButton").GetComponent<Button>();
        AgentHandTransform = transform.Find("AgentHand");
        ScoreText = AgentHandTransform.Find("Title").Find("ScoreText").GetComponent<TextMeshProUGUI>();
        AgentCardCountText = AgentHandTransform.Find("Image").Find("CardCountText").GetComponent<TextMeshProUGUI>();
        ConfidenceText = AgentHandTransform.Find("ConfidenceText").GetComponent<TextMeshProUGUI>();
        ScoreText.text = string.Format("Human: {0} \t\t\t\t\t\t\tComputer: {1}", playerScore, computerScore);
        ConfidenceText.text = "";
        AgentDelayTimeout = 0.5f;

        randomAgent = new RandomAgent();
        greedyAgent = new GreedyAgent();
        learnedAgent = new LearnedAgent(3);
    }
    
    void Update()
    {
        if(AgentIsDelaying) {
            AgentDelayTimer -= Time.deltaTime;
            if (AgentDelayTimer < 0) {
                AgentIsDelaying = false;
                AgentTakeTurn();
            }
        }
    }
}
