using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class PopupScript : MonoBehaviour
{
    
    private float timeOut = 4f;
    public float timer;
    private Image bg;
    private TextMeshProUGUI text;
    public Color dilateColor = Color.black;
    // Start is called before the first frame update
    void Start()
    {
        timer = timeOut;
        bg = gameObject.GetComponent<Image>();
        text = gameObject.transform.Find("Text").GetComponent<TextMeshProUGUI>();
    }

    public void setText(string Text) {
        gameObject.transform.Find("Text").GetComponent<TextMeshProUGUI>().text = Text;
    }
    public void setText(string Text, Color color) {
        setText(Text);
        this.dilateColor = color;
    }

    // Update is called once per frame
    void Update()
    {
        timer -= Time.deltaTime;
        float alpha = 2 * Mathf.Min(timer / timeOut, .5f);
        bg.color = new Color(bg.color.r, bg.color.g, bg.color.b, alpha);
        text.color = new Color(dilateColor.r, dilateColor.g, dilateColor.b, alpha);
        if (timer < 0) Destroy(gameObject);
    }
}
