using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine;
using UnityEngine.UI;

public class CardButton 
{
    public Image bg;
    public Button bt;
    public Image btimg;
    private static Color SelectedColor = new Color(0, .5f, 1);

    public static CardButton NewCardButton(string card, Action<int> action = null, int actionNum = 0) {
        CardButton cb = new CardButton();
        cb.bg = GameObject.Instantiate(Resources.Load<Image>("Card"));
        Transform t = cb.bg.transform.Find("Button");
        cb.bt = t.GetComponent<Button>();
        cb.btimg = cb.bt.GetComponent<Image>();
        t.GetComponent<Image>().sprite = Resources.Load<Sprite>(Path.Combine("Cards", card));
        if (action != null) {
            cb.bt.onClick.AddListener(delegate {action(actionNum);});
        }
        return cb;
    }

    public void SetSelected() { btimg.color = SelectedColor; }
    public void SetUnselected() { btimg.color = Color.white; }

}
