using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using TensorFlow;

public class LearnedAgent : Agent
{
    public TFGraph graph;
    private TextAsset graphModel;
    private TFSession session;
    private string modelName;

    private float[,] RunModel(float[,,] input) {
        var runner = session.GetRunner();
        runner.AddInput(graph[modelName + "/Placeholder"][0], input);
        runner.Fetch(graph[modelName + "_2/output/Sigmoid"][0]);
        float[,] output = runner.Run()[0].GetValue() as float[,];
        return output;
    }

    public LearnedAgent(string modelName) {
        name = "AI";
        this.modelName = modelName;
        graph = new TFGraph();
        graphModel = Resources.Load<TextAsset>("opt_" + modelName);
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

        //float[,,] input = new float[1, 5, 60];
        //for (int i = 0; i < 5; i += 2)
        //    for (int j = 0; j < 60; j++)
        //        input[0, i, j] = 1;

        float[,,] input = new float[legalMoves.Count, 5, 60];
        for (int i = 0; i < legalMoves.Count; i++) {
            for (int j = 0; j < 15; j++) {
                input[i, 4-histAg[j], j] = 1;
                input[i, 4-histOp[j], j + 15] = 1;
                input[i, 4-hand[j] + legalMoves[i][j], j + 30] = 1;
                input[i, 4-legalMoves[i][j], j + 45] = 1;
            }
        }



        float[,] output = RunModel(input);
        //Debug.Log(output.GetLength(0) + ", " + output.GetLength(1));
        float maxScore = -1;
        int maxIndex = 0;
        for(int i=0; i<output.GetLength(0); i++) {
            //Debug.Log(output[i, 0]);
            if(output[i, 0] > maxScore) {
                maxScore = output[i, 0];
                maxIndex = i;
            }
        }

        confidence = maxScore;
        return new Tuple<int[], float>(legalMoves[maxIndex], maxScore);
    }
}
