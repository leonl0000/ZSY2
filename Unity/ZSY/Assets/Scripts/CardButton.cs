using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class CardButton 
{
    public Image bg;
    public Button bt;
    public Image btimg;
    private static Color SelectedColor = new Color(0, .5f, 1);

    public static CardButton NewCardButton(string card, Action<int, bool> action = null, int actionNum = 0) {
        CardButton cb = new CardButton();
        cb.bg = GameObject.Instantiate(Resources.Load<Image>("Card"));
        Transform t = cb.bg.transform.Find("Button");
        cb.bt = t.GetComponent<Button>();
        cb.btimg = cb.bt.GetComponent<Image>();
        t.GetComponent<Image>().sprite = Resources.Load<Sprite>(Path.Combine("Cards", card));
        if (action != null) {
            EventTrigger trigger = cb.bt.gameObject.AddComponent<EventTrigger>();
            var pointerDown = new EventTrigger.Entry();
            pointerDown.eventID = EventTriggerType.PointerDown;
            pointerDown.callback.AddListener(delegate {
                if(cb.bt.enabled) action(actionNum, cb.flipSelected());
            });
            trigger.triggers.Add(pointerDown);
            var pointerEnter = new EventTrigger.Entry();
            pointerEnter.eventID = EventTriggerType.PointerEnter;
            pointerEnter.callback.AddListener(delegate {
                if (Input.GetMouseButton(0) && cb.btimg.color == Color.white && cb.bt.enabled)                   
                    action(actionNum, cb.flipSelected());
            });
            trigger.triggers.Add(pointerEnter);
        }
        return cb;
    }

    public bool flipSelected() {
        if (btimg.color == SelectedColor) {
            btimg.color = Color.white;
            return false;
        }
        btimg.color = SelectedColor;
        return true;
    }
    public bool getSelected() { return btimg.color == SelectedColor; }
    public void SetSelected() { btimg.color = SelectedColor; }
    public void SetUnselected() { btimg.color = Color.white; }

}
