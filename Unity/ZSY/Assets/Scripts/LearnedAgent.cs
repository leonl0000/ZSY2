using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using TensorFlow;

#if UNITY_EDITOR
public class LearnedAgent : Agent
{
    public TFGraph graph;
    private TextAsset graphModel;
    private TFSession session;
    private int kindIndex;

    private readonly static string[] agentFiles = {
        "opt_DenseNet3",
        "opt_ConvNet2",
        "opt_ComboAgent_Mean_3",
        "opt_ComboAgent_Min_9"
    };
    private readonly static string[][] agentInputs = {
        new string[] { "DenseNet3" },
        new string[] { "ConvNet2" },
        new string[] { "ConvNet2", "ConvNet3", "ConvNet1"},
        new string[] {"ConvNet2", "ConvNet3", "ConvNet1", "ConvNet4", "ConvNet12", "ConvNet9", "ConvNet6", "ConvNet7", "ConvNet10"}
    };
    private readonly static string[] outputNames = {
        "DenseNet3_2/output/Sigmoid",
        "ConvNet2_2/output/Sigmoid",
        "Mean",
        "Min",
    };


    private float[] RunModel(float[,,] input) {
        var runner = session.GetRunner();
        foreach(string s in agentInputs[kindIndex])
            runner.AddInput(graph[s + "/Placeholder"][0], input);
        runner.Fetch(graph[outputNames[kindIndex]][0]);
        if (kindIndex < 2) {
            float[,] temp = runner.Run()[0].GetValue() as float[,];
            float[] output = new float[temp.GetLength(0)];
            return output;
        } else return runner.Run()[0].GetValue() as float[];
    }

    public LearnedAgent(int kindIndex) {
        name = "AI";
        this.kindIndex = kindIndex;
        graph = new TFGraph();
        graphModel = Resources.Load<TextAsset>(agentFiles[kindIndex]);
        graph.Import(graphModel.bytes);
        session = new TFSession(graph);
    }

    public override Tuple<int[], float> GetMove(int[] hand, List<int[]> legalMoves, List<int[]> history) {
        int[] histAg = new int[15];
        int[] histOp = new int[15];
        for(int i=history.Count-1; i>-1; i--) {
            for (int j = 0; j < 15; j++)
                histOp[j] += history[i][j];
            i--;
            if (i < 0) break;
            for (int j = 0; j < 15; j++)
                histAg[j] += history[i][j];
        }
        Debug.Log(string.Join(", ", histAg));
        Debug.Log(string.Join(", ", histOp));

        float[,,] input = new float[legalMoves.Count, 5, 60];
        for (int i = 0; i < legalMoves.Count; i++) {
            for (int j = 0; j < 15; j++) {
                input[i, 4-histAg[j], j] = 1;
                input[i, 4-histOp[j], j + 15] = 1;
                input[i, 4-hand[j] + legalMoves[i][j], j + 30] = 1;
                input[i, 4-legalMoves[i][j], j + 45] = 1;
            }
        }



        float[] output = RunModel(input);
        float maxScore = -1;
        int maxIndex = 0;
        for(int i=0; i<output.Length; i++) {
            if(output[i] > maxScore) {
                maxScore = output[i];
                maxIndex = i;
            }
        }

        confidence = maxScore;
        return new Tuple<int[], float>(legalMoves[maxIndex], maxScore);
    }
}
#endif
