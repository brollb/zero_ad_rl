def graph(file_name):
    # Parse the data file
    contents = []
    ys = []
    xs = []
    print(file_name)
    with open(f'outputs/{file_name}','r') as f:
        try:
            contents = f.readlines()
        except:
            raise Exception("Couldn't read the file")

    # grab numbers
    for line in contents:
        try: # Split on reward:
            ys.append(float(line.split('reward: ')[1].replace('\n','')))
        except:
            pass

    offset = int(len(ys)*.02)

    # If offset is zero, no data
    if offset == 0:
        raise Exception("Offset is zero")

    avgys = []
    for i in range(0,len(ys),offset):
        total = 0
        for j in range(i, i+offset):
            if j < len(ys):
                total+=ys[j]
        avgys.append(total/offset)

    xs = [i for i in range(len(avgys))]
    import matplotlib.pyplot as plt

    plt.subplot(1, 1, 1)
    plt.plot(xs,avgys)
    plt.ylabel('Reward mean')
    plt.xlabel('Time')
    plt.title('Reward mean vs time')
    plt.show()

if __name__ == '__main__':
    import os
    files = None
    try:
        files = os.listdir("outputs")
    except:
        raise Exception("Couldn't find directory outputs")
    
    print("Which file to graph:")
    for i in range(len(files)):
        print(f"\t{i+1} {files[i]}")

    which = int(input("\nChoice: "))
    graph(files[which-1])

