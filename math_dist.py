import sys

def correctness_check(N, aggs):
	print("CHECK START")
	all_aggs = []
	for i in range(N):
		for j in range(i+1, N):
			for k in range(j+1, N):
				all_aggs.append([i, j, k])

	sorted_aggs = []
	for g in aggs:
		for agg in g:
			sorted_aggs.append(sorted(agg))
	sorted_aggs.sort()
	
	not_found = []
	for agg in all_aggs:
		if agg in sorted_aggs:
			sorted_aggs.remove(agg)
		else:
			not_found.append(agg)
	
	if len(not_found) == 0 and len(sorted_aggs) == 0:
		print("CORRECT")
	elif len(not_found) > 0:
		print("NOT FOUND")
		print(not_found)
	elif len(sorted_aggs) > 0:
		print("TOO MUCH")


def even_dist_aggs(N):
	if N == 3:
		return [[[0, 1, 2]]]

	aggs = [[] for _ in range(N)]

	distances = []

	# Create first shift patterns
	curr = 1
	while (True):
		dis = [curr, curr, N - (curr * 2)]
		if N - (curr * 2) < curr:
			break
		while dis[2] > dis[0]:
			distances.append(dis.copy())
			aggs[0].append([0, dis[0], dis[0] + dis[1]])
			dis[1] += 1
			dis[2] -= 1
		curr += 1

	#print(distances)
	#print()

	total_agg_num = N * (N-1) * (N-2) / 6
	leftover_agg = total_agg_num % N
	if leftover_agg != 0:
		extra_pattern = [0, int(N / 3), int(N / 3) * 2]

	for i in range(1, N):
		for p in aggs[0]:
			aggs[i].append([(p[0] + i) % N, (p[1] + i) % N, (p[2] + i) % N])

	for i in range(int(leftover_agg)):
		aggs[i].append([extra_pattern[0] + i, extra_pattern[1] + i, extra_pattern[2] + i])

	return aggs

def two_col_templates(N, front):
	nc2 = N * (N-1) / 2
	aggs_per_col = int(nc2 / N)
	ret = [[] for _ in range(N)]
	for i in range(N):
		agg_range = aggs_per_col + 1
		if N % 2 == 0:
			if i < N / 2 and front:
				agg_range += 1
			elif i >= N / 2 and not front:
				agg_range += 1
		for agg in range(1, agg_range):
			ret[i].append([i, (i + agg) % N])
	return ret


def reduce(N, reduce_col_num):
	aggs = even_dist_aggs(N-reduce_col_num)
	if reduce_col_num == 1:
		tct = two_col_templates(N - 1, False)
		for i in range(N-1):
			for t in tct[i]:
				aggs[i].append([t[0], t[1], N - 1])
	elif reduce_col_num == 2:
		for i in range(N-2):
			aggs[i].append([i, N - 2, N - 1])
		tct = two_col_templates(N - 2, False)
		for i in range(N-2):
			for t in tct[i]:
				aggs[i].append([t[0], t[1], N - 2])
		tct = two_col_templates(N - 2, True)
		for i in range(N-2):
			for t in tct[i]:
				aggs[i].append([t[0], t[1], N - 1])
	return aggs


def shift_aggs(aggs, shift):
	for group in aggs:
		for agg in group:
			agg[0] += shift
			agg[1] += shift
			agg[2] += shift


def split(N, split):
	aggs_front = even_dist_aggs(split)
	for col in range(split):
		for i in range(split, N-1):
			for j in range(i + 1, N):
				aggs_front[col].append([col, i, j])
	tct_front = two_col_templates(split, True)
	tct_back = two_col_templates(split, False)
	curr = 0
	for col in range(split, N):
		curr_tct = tct_back
		if curr % 2 == 1:
			curr_tct = tct_front
		for i in range(split):
			for t in curr_tct[i]:
				aggs_front[i].append([t[0], t[1], col])
		curr += 1
	
	return aggs_front


def even_dist_dpu(N, R):
	if N % R == 0:
		ret = []
		for i in range(int(N/R)):
			split_aggs = split(N - R*i, R)
			shift_aggs(split_aggs, R*i)
			if len(split_aggs[0]) != 0:
				ret.append([1, split_aggs])
		return ret
	elif R % N == 0:
		return [[int(R/N), even_dist_aggs(N)]]
	if N == 3:
		return [[R, even_dist_aggs(N)]]
	if N > R:
		if N - 1 == R:
			return [[1, reduce(N, 1)]]
		elif N - 2 == R:
			return [[1, reduce(N, 2)]]
		else:
			split_aggs = split(N, R)
			aggs_back = even_dist_dpu(N-R, R)
			for a in aggs_back:
				shift_aggs(a[1], R)
			return [[1, split_aggs]] + aggs_back
	elif N < R:
		if R % (N - 1) == 0:
			return [[int(R/(N-1)), reduce(N, 1)]]
		elif R % (N-2) == 0:
			return [[int(R/N-2), reduce(N, 2)]]
		else:
			split_col = R
			while (split_col > N):
				split_col = int(split_col / 2)
			split_aggs = split(N, split_col)
			aggs_back = even_dist_dpu(N - split_col, R)
			for a in aggs_back:
				shift_aggs(a[1], split_col)
			return [[int(R / split_col), split_aggs]] + aggs_back





	