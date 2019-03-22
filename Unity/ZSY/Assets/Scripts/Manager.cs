using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;



public class Manager : MonoBehaviour
{
    public GameObject cv;
    public GameObject cv2;
    public GameObject PostGamePanel;
    public GameObject OpeningScreen;
    public GameObject SettingsScreen;
    private GameObject AboutScreen;
    public Slider thinkSlider;
    public TextMeshProUGUI thinkText;
    public TextMeshProUGUI confidenceToggleText;
    public zsyGame zsy;
    public TextMeshProUGUI postGameText;

    public void Popup(string text) {
        GameObject g = Instantiate(Resources.Load<GameObject>("Popup"));
        g.transform.SetParent(cv2.transform, false);
        g.GetComponent<PopupScript>().setText(text);
    }

    public void Popup(string text, Color color) {
        GameObject g = Instantiate(Resources.Load<GameObject>("Popup"));
        g.transform.SetParent(cv2.transform, false);
        g.GetComponent<PopupScript>().setText(text, color);
    }

    public void tutorialButton() {
        zsy.StartTutorialGame();
        OpeningScreen.SetActive(false);
    }
    public void playDQNButton() {
        zsy.StartGame(zsy.learnedAgent);
        OpeningScreen.SetActive(false);
    }
    public void playGreedyButton() {
        zsy.StartGame(zsy.greedyAgent);
        OpeningScreen.SetActive(false);
    }
    public void playRandomButton() {
        zsy.StartGame(zsy.randomAgent);
        OpeningScreen.SetActive(false);
    }

    public void settingsButton() {
        SettingsScreen.SetActive(true);
        thinkSlider.value = zsy.AgentDelayTimeout;
        thinkText.text = string.Format("I'll pretend to think for {0:0.0} seconds before making my move", zsy.AgentDelayTimeout);
    }
    public void closeSettings () { SettingsScreen.SetActive(false); }
    public void sliderValueChange(Slider s) {
        zsy.AgentDelayTimeout = s.value;
        thinkText.text = string.Format("I'll pretend to think for {0:0.0} seconds before making my move", s.value);
    }
    public void confidenceToggle(Toggle t) {
        zsy.showConfidence = t.isOn;
        confidenceToggleText.text = zsy.showConfidence ? "I'll tell you how confident I am" : "I'll keep my confidence to myself";
    }


    public void playAgain() {
        zsy.StartGame();
        PostGamePanel.SetActive(false);
    }

    public void mainMenu() {
        OpeningScreen.SetActive(true);
        PostGamePanel.SetActive(false);
        zsy.quitTutorial();
        zsy.playerScore = 0;
        zsy.computerScore = 0;
        zsy.setScoreText();
    }

    public void postGame(bool win) {
        PostGamePanel.SetActive(true);
        postGameText.text = (win ? "YOU WIN!" : "Sorry, I win this time.") + "\nPlay Again?";
    }

    public void aboutButton(bool on) {
        AboutScreen.SetActive(on);
    }

    public void quitButton() {
        if (OpeningScreen.active) Application.Quit();
        else mainMenu();
    }

    private int prevW, prevH;

    void Start() {
        //Screen.SetResolution(1280, 720, false);
        prevW = Screen.width;
        prevH = Screen.height;
        cv = GameObject.Find("Canvas");
        zsy = cv.GetComponent<zsyGame>();
        cv2 = GameObject.Find("Canvas2");
        if (cv2 != null) {
            PostGamePanel = cv2.transform.Find("PostGamePanel").gameObject;
            postGameText = PostGamePanel.transform.Find("Text").GetComponent<TextMeshProUGUI>();
            PostGamePanel.SetActive(false);
            AboutScreen = cv2.transform.Find("AboutScreen").gameObject;
            AboutScreen.transform.SetAsLastSibling();
            AboutScreen.SetActive(false);
            OpeningScreen = cv2.transform.Find("OpeningScreen").gameObject;
            SettingsScreen = cv2.transform.Find("SettingsScreen").gameObject;
            thinkSlider = SettingsScreen.transform.Find("Slider").GetComponent<Slider>();
            thinkText = SettingsScreen.transform.Find("ThinkText").GetComponent<TextMeshProUGUI>();
            confidenceToggleText = SettingsScreen.transform.Find("Toggle").Find("Text").GetComponent<TextMeshProUGUI>();
            confidenceToggleText.text = zsy.showConfidence ? "I'll tell you how confident I am" : "I'll keep my confidence to myself";
            SettingsScreen.transform.SetAsLastSibling();
            SettingsScreen.SetActive(false);
        }



        //text.text = "" + a[0, 0] + "\n" + Application.dataPath + "\n" + Application.persistentDataPath;

    }

    public void gotoReport() {
        Application.OpenURL("https://github.com/leonl0000/ZSY2/blob/master/Writeups/Final%20Report.pdf");
    }

    // Update is called once per frame
    private const float aspectLock = 9f/16;
    bool isResizing;
    private const float resizeTimeout = 1f;
    private float resizeCountdown;
    void Update()
    {
        //if ((Screen.width != prevW || Screen.height != prevH)) {
        //    isResizing = true;
        //    prevW = Screen.width;
        //    prevH = Screen.height;
        //} else if (isResizing) {
        //    isResizing = false;
        //    resizeCountdown = resizeTimeout;
        //} else if (resizeCountdown > 0) {
        //    resizeCountdown -= Time.deltaTime;
        //    if (resizeCountdown < 0) {
        //        Screen.SetResolution(Screen.width, (int)(Screen.width * aspectLock), Screen.fullScreen);
        //        prevW = Screen.width;
        //        prevH = Screen.height;
        //    }
        //}
        //if (Input.GetKeyDown(KeyCode.Escape)) Screen.fullScreen = false;
            
    }
}
