
from matplotlib import pyplot as plt
import numpy as np
import wavelets
from tqdm import tqdm
import seaborn as sns; sns.set()


# Load dataset
def load_data(fname):
	fdata = open(fname)
	# Skip 4 header lines
	for _ in range(4):
		fdata.readline()
	# Init data lists
	acc_x = []
	acc_y = []
	acc_z = []
	# Populate, take only accelerometer right thig
	for line in fdata:
		line_int = [int(x) for x in line.split("\t")]
		acc_x.append(line_int[12])
		acc_y.append(line_int[13])
		acc_z.append(line_int[14])
	# Convert to numpy array and return
	return np.array(acc_x), np.array(acc_y), np.array(acc_z)


#plt.figure(1)
def plot_data(acc_x, acc_y, acc_z):
	plt.plot(acc_x, label="acc_x")
	plt.plot(acc_y, label="acc_y")
	plt.plot(acc_z, label="acc_z")
	plt.xlim((0,100))
	plt.legend()
	plt.show()
	#plt.savefig("aa.png", dpi=300)
	#plt.clf()


"""
STEPS:

 - Given sequence of human steps, isolate single step
 	--> Time series segmentation
 - choose as baseline the step that minimizes sum of the distance between others
 - evaluate
"""





#W = 10  #max displacement
SIZE = 100
DTW = np.zeros((SIZE,SIZE), dtype=np.int64)
#dist_func = lambda x, y: np.abs(x - y)
dist_func = lambda x, y: (x-y)**2
def dist(u,v):
	"""
	Calculates the distance between two signals, using time warp tecnique
	"""
	global DTW
	assert(len(v)<SIZE and len(u)<SIZE)
	#W = max(W, abs(len(u), len(v)))

	DTW += 10**9
	DTW[-1: ] = 0
	DTW[: -1] = 0

	for i, ui in enumerate(u):
		for j, vj in enumerate(v):
			cost = dist_func(ui, vj)
			DTW[i,j] = cost + min(DTW[i-1,j], DTW[i,j-1], DTW[i-1,j-1])

	#sns.heatmap(DTW)
	#plt.show()
	#plt.clf()

	return DTW[len(u)-1, len(v)-1]




def extract_steps(data, x_start, x_end):
	step_width_min = 40
	step_width_max = 60

	step_original = data[x_start:x_end]
	step_prev = data[x_start:x_end]
	errors = np.zeros(len(data), dtype=np.int64)

	x_start = x_end
	steps = [x_start]   #x where the steps start
	stop = False
	while not stop:
		min_err = np.inf
		min_x = -1
		for x in range(x_start+step_width_min, x_start+step_width_max):
			if x >= len(data):
				stop = True
				break
			#err = dist(data[x_start:x], step_prev)
			err = dist(data[x_start:x], step_original)
			errors[x] = err
			if err < min_err:
				min_err = err
				min_x = x
		else:
			step_prev = data[x_start:min_x]
			x_start = min_x
			steps.append(x_start)
			print("K")

	# Print starting of a step
	for i, x in enumerate(steps):
		print(f"step {i} starting at {x}")

	return steps, errors



def plot_bunch_steps(data, x_steps, smooth=True, n=10):
	steps = get_step_from_xstep(data, x_steps, smooth=smooth)
	for step in steps[:n]:
		plt.plot(step)
	#plt.show()
	plt.savefig(f"bunch_of_steps_{PR}_{('smooth' if smooth else 'raw')}.png", dpi=300)
	plt.clf()


# Plotting mean length of a step
def plot_step_mean_length(steps):
	lengths = []
	for i in range(1,len(steps)):
		lengths.append(steps[i]-steps[i-1])
		print(f"step {i-1} length: {lengths[-1]}")
	plt.plot(lengths)
	plt.show()


def plot_errors(errors):
	error_mean = np.mean(errors[400:1400])
	print("errors mean", error_mean)
	plt.plot(errors)
	plt.hlines(error_mean, 0, 2000)
	plt.ylim((error_mean*0.8,error_mean*1.2))
	plt.xlim((400,1400))
	plt.show()
	#plt.savefig(f"window_size_{WINDOW_SIZE}.png", dpi=300)


def pipeline(data, x_start, x_end):
	steps, errors = extract_steps(data, x_start, x_end)
	#plot_bunch_steps(data, steps, smooth=True)
	#plot_bunch_steps(data, steps, smooth=False)
	#plot_step_mean_length(steps)
	#plot_errors(errors)
	return steps


def get_step_from_xstep(data, x_steps, smooth=True):
	steps = []
	for i in range(len(x_steps)-1):
		step = data[x_steps[i]:x_steps[i+1]]
		if smooth:
			step = wavelets.wavelet_smooth(step)
		steps.append(step)
	return steps

def get_central_step(data, x_steps):
	steps = get_step_from_xstep(data, x_steps)
	print("Calculating pairwise distances...")
	distance_sums = []
	for s1 in tqdm(steps):
		distance_sum = 0
		for s2 in steps:
			distance_sum += dist(s1,s2)
		distance_sums.append(distance_sum)

	# Plot the sum of the distances for each step
	#plt.scatter(range(len(distance_sums)), sorted(distance_sums))
	#plt.show()

	return steps[np.argmin(distance_sums)]


def train_eval_split(data):
	return data[:1450], data[1450:2000]


# Dictionary giving x_start and x_end of a step for each file
first_step = {
	"HuGaDB/HuGaDB_v1_walking_01_00.txt": (23, 70),
	"HuGaDB/HuGaDB_v1_walking_03_00.txt": (137, 190),
	"HuGaDB/HuGaDB_v1_walking_04_00.txt": (268, 324),  #has long step
}
first_step_eval = {
	"HuGaDB/HuGaDB_v1_walking_01_00.txt": (1564, 1611),
	"HuGaDB/HuGaDB_v1_walking_03_00.txt": (1144, 1197)
	#"HuGaDB/HuGaDB_v1_walking_03_01.txt": (1469, 1550)
}

if __name__ == '__main__':
	pr_cnt_list = [("01", "00"), ("03", "00")]
	baseline_steps = []
	for PR, CNT in pr_cnt_list:
		fname = f"HuGaDB/HuGaDB_v1_walking_{PR}_{CNT}.txt"
		acc_x, acc_y, acc_z = load_data(fname)
		#plot_data(acc_x, acc_y, acc_z)
		data = acc_y[:1100]
		x_start, x_end = first_step[fname]
		
		x_steps = pipeline(data, x_start, x_end)

		baseline_step = get_central_step(data, x_steps)
		#plt.figure(2)
		#plt.plot(baseline_step)
		#plt.show()

		baseline_steps.append(baseline_step)

	# Get test steps
	steps = [[], []]
	for i, (PR, CNT) in enumerate(pr_cnt_list):
		fname = f"HuGaDB/HuGaDB_v1_walking_{PR}_{CNT}.txt"
		acc_x, acc_y, acc_z = load_data(fname)
		#plot_data(acc_x, acc_y, acc_z)
		data = acc_y
		x_start, x_end = first_step_eval[fname]

		x_steps = pipeline(data, x_start, x_end)
		steps[i] = get_step_from_xstep(data, x_steps)[:10]  #too many

	# Plot distance of test step with baseline steps
	plt.clf()
	for i, baseline_step in enumerate(baseline_steps):
		for j, steps_j in enumerate(steps):
			cost_sum = 0
			for step in steps_j:
				cost = dist(step, baseline_step) / len(step)
				cost_sum += cost
				plt.plot([0], [cost], "bx" if i==j else "rx")
		plt.xticks([])
		#plt.savefig(f"clf_{i}.png", dpi=300)
		plt.show()
		plt.clf()

