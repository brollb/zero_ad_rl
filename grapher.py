# Parse the data file
contents = []
ys = []
xs = []
with open('outputs/20200412_runaway.output','r') as f:
    contents = f.readlines()

# grab numbers
for line in contents:
    try: # Split on reward:
        ys.append(float(line.split('reward: ')[1].replace('\n','')))
    except:
        pass

offset = int(len(ys)*.02)
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
plt.ylabel('Average distance')
plt.xlabel('Episode')
plt.title('First chart')
plt.savefig('test1.png')
plt.show()
