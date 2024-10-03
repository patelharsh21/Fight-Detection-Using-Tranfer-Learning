def findCapableWinners(pta, ptb, ptc):
    n = len(pta)
    players_boosters = []

    # Prepare the power boosters of each player and sort them in descending order
    for i in range(n):
        players_boosters.append(sorted([pta[i], ptb[i], ptc[i]], reverse=True))

    capable_winners = 0

    # Check for each player whether they can defeat all others
    for i in range(n):
        capable = True
        for j in range(n):
            if i != j:
                win_count = sum(players_boosters[i][k] > players_boosters[j][k] for k in range(3))
                if win_count < 2:
                    capable = False
                    break
        if capable:
            capable_winners += 1

    return capable_winners

# Example usage:
pta = [9, 5, 11]
ptb = [4, 12, 3]
ptc = [2, 10, 13]

result = findCapableWinners(pta, ptb, ptc)
print(result)  # Output should be the number of capable winners
