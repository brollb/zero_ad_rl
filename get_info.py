from statistics import mean, median, mode, variance, stdev
import sys
import operator

checkpoint = 1
data = []

def print_stats(wins, defeats, values):
    global checkpoint, data

    values.sort()
    print("########## CHECKPOINT " + str(checkpoint) + " ##########")
    print("Total Games: " + str((wins+defeats)))
    print("Wins: " + str(wins))
    print("Defeats: " + str(defeats))
    print("Best Reward: " + str(values[-1]))
    print("Worst Reward: " + str(values[0]))
    print("Variance: " + str(variance(values)))
    print("Std Deviation: " + str(stdev(values)))
    print("Mean Reward: " + str(mean(values)))
    print("Median Reward: " + str(median(values)))
    try:
        print("Reward Mode: " + str(mode(values)))
    except:
        print("No best representation for the mode found.")
    print("#################################\n")
    
    data.append((checkpoint, wins, defeats, values[-1], values[0], variance(values), stdev(values), mean(values), median(values)))

    checkpoint += 1



def print_all_stats(wins, defeats, values):
    global data

    values.sort()
    
    print("########## FINAL CHECKPOINT ##########")
    print("Total Games: " + str((wins+defeats)))
    print("Wins: " + str(wins))
    print("Defeats: " + str(defeats))
    print("Best Reward: " + str(values[-1]))
    print("Worst Reward: " + str(values[0]))
    print("Variance: " + str(variance(values)))
    print("Std Deviation: " + str(stdev(values)))
    print("Mean Reward: " + str(mean(values)))
    print("Median Reward: " + str(median(values)))
    try:
        print("Reward Mode: " + str(mode(values)))
    except:
        print("No best representation for the mode found.")
    print("#################################\n")
    
    data.sort(key = operator.itemgetter(1), reverse = True)
    print("Checkpoint with the most wins: " + str(data[0][0]))
    print(data[0])
    print("")

    data.sort(key = operator.itemgetter(3), reverse = True)
    print("Checkpoint with the best reward: " +str(data[0][0]))
    print(data[0])
    print("")

    data.sort(key = operator.itemgetter(4), reverse = True)
    print("Checkpoint with the worst reward: " + str(data[0][0]))
    print(data[0])
    print("")




if __name__ == "__main__":
    try:
        checkpoint_value = int(sys.argv[1])
    except:
        print("Checkpoint value needed.")

    num = 0
    checkpoint_wins = 0
    checkpoint_defeats = 0
    checkpoint_values = []

    all_wins = 0
    all_defeats = 0
    all_values = []

    #TODO dont hard code file name in here.
    with open("a.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                info = line.split()[-1]

                if info == "won":
                    checkpoint_wins += 1
                elif info == "defeated":
                    checkpoint_defeats += 1
                else:
                    checkpoint_values.append(float(info))
                
                num += 1

            except:
                #TODO Print something meaningful here
                pass

            if num == checkpoint_value:
                num = 0
                all_wins += checkpoint_wins
                all_defeats += checkpoint_defeats
                all_values.extend(checkpoint_values)

                print_stats(checkpoint_wins, checkpoint_defeats, checkpoint_values)

                checkpoint_wins = 0
                checkpoint_defeats = 0
                checkpoint_values = []

        print_all_stats(all_wins, all_defeats, all_values)

