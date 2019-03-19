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

public class TutorialAgent : Agent
{
    public TutorialAgent() { name = "Tutorial"; counter = -1; }
    private static readonly int[] emptyMove = new int[15];
    private short counter;
    private static readonly int[][] staticMoves = new int[][] {
        new int[15] {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
        new int[15] {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
        new int[15] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        new int[15] {0,0,0,0,0,0,3,0,0,0,0,0,0,0,0},
        new int[15] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        new int[15] {0,0,0,0,0,0,0,0,0,2,2,0,0,0,0},
        new int[15] {2,3,2,0,0,0,0,0,0,0,0,0,0,0,0},
        new int[15] {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        };
    public override Tuple<int[], float> GetMove(int[] hand, List<int[]> legalMoves, List<int[]> history) {
        counter += 1;
        return new Tuple<int[], float>(staticMoves[counter], 0f);
    }
}
