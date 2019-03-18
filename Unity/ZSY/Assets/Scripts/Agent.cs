using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using Random = UnityEngine.Random;



public abstract class Agent
{
    public string name = "Abstract";
    public float confidence = 0;
    public abstract Tuple<int[], float> GetMove(int[] hand, List<int[]> legalMoves, List<int[]> history);
}

public class RandomAgent : Agent
{
    public RandomAgent() { name = "Random";}
    public override Tuple<int[], float> GetMove(int[] hand, List<int[]> legalMoves, List<int[]> history) {
        return new Tuple<int[], float>(legalMoves[Random.Range(0, legalMoves.Count)], 0f);
    }
}

public class GreedyAgent : Agent
{
    public GreedyAgent() { name = "Greedy"; }
    private static readonly int[] emptyMove = new int[15];
    public override Tuple<int[], float> GetMove(int[] hand, List<int[]> legalMoves, List<int[]> history) {
        int[] move;
        if (legalMoves.Count == 1 || !legalMoves[0].SequenceEqual(emptyMove)) move = legalMoves[0];
        else move = legalMoves[1];
        return new Tuple<int[], float>(move, 0f);
    }
}
