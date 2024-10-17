import numpy as np
import copy
from PIL import Image

def calculate_score(ppm, mode):
  if mode == 0:
    return np.average(ppm.flatten())
  elif mode == 1:
    return np.var(ppm.flatten())
  elif mode == 2:
    ppm_red = ppm[:,:,0]
    return np.average(ppm_red)

def generate_pixel_data(interaction_matrix, redblue=True):
	original_shape = interaction_matrix.shape
	total_counts_values = interaction_matrix.flatten().sum()
	average_frequency_per_value_pair = total_counts_values / (original_shape[0] * original_shape[1])
	img_flat = interaction_matrix.flatten()
	ret = np.multiply(img_flat, 256 / (2 * average_frequency_per_value_pair)).astype(np.uint32)
	#ret = np.clip(ret, a_min=0, a_max=255).astype(np.int16)
	ret = np.multiply(ret, 2).astype(np.int16)
	ret = np.subtract(ret, 255)
	# Set to False for red-green scheme
	if redblue:
		ret = np.array([ret, np.zeros(original_shape[0] * original_shape[1]).astype(np.int16), ret * -1])
	else:
		ret = np.array([ret, ret * -1, np.zeros(original_shape[0] * original_shape[1]).astype(np.int16)])
	ret = np.clip(ret, a_min = 0, a_max=255).astype(np.uint8).T
	ret = ret.reshape(original_shape[0], original_shape[1], -1)
	return ret

def swap_agg_axes(agg, curr, goal):
	curr_order = copy.deepcopy(curr)
	for g in goal:
		g_index = goal.index(g)
		c_index = curr_order.index(g)
		curr_order[g_index], curr_order[c_index] = curr_order[c_index], curr_order[g_index]
		if c_index != g_index:
			agg = np.swapaxes(agg, c_index, g_index)
	return agg


def find_ppms(axes, aggs, agg_info, frames):
	found = False
	for i, cols in enumerate(agg_info):
		if set(axes) == set(cols):
			target_agg = aggs[i]
			target_agg_info = cols
			found = True

	if not found:
		print(axes)
		print("Columns not found")
		return []
	else:
		ret = []
		target_agg = swap_agg_axes(target_agg, target_agg_info, axes)
		#display(target_agg.shape)
		target_agg_dim0 = target_agg.shape[0]
		buckets_per_frame = int((target_agg_dim0 + frames - 1) / frames)
		#target_agg = target_agg.reshape(bucket_num, bucket_num ** 2)
		#target_agg = target_agg.reshape([target_agg.shape[0], -1])
		for f in range(frames):
			frame_agg = target_agg[buckets_per_frame * f : buckets_per_frame * (f+1)]
			#display(frame_agg.shape)
			frame_agg = np.sum(frame_agg, axis = 0)

			# Reverse frame agg on axis 0 so that y axis is in increasing order
			frame_agg = frame_agg[::-1]
			ppm = generate_pixel_data(frame_agg)

			ret.append(ppm)
		return ret

def find_and_display_aggs(axes, aggs, agg_info, frames):
	found = False
	for i, cols in enumerate(agg_info):
		if set(axes) == set(cols):
			target_agg = aggs[i]
			target_agg_info = cols
			found = True

	if not found:
		print("Columns not found")
	else:
		target_agg = swap_agg_axes(target_agg, target_agg_info, axes)
		display(target_agg.shape)
		target_agg_dim0 = target_agg.shape[0]
		buckets_per_frame = int((target_agg_dim0 + frames - 1) / frames)
		#target_agg = target_agg.reshape(bucket_num, bucket_num ** 2)
		#target_agg = target_agg.reshape([target_agg.shape[0], -1])
		for f in range(frames):
			frame_agg = target_agg[buckets_per_frame * f : buckets_per_frame * (f+1)]
			#display(frame_agg.shape)
			frame_agg = np.sum(frame_agg, axis = 0)

			# Reverse frame agg on axis 0 so that y axis is in increasing order
			frame_agg = frame_agg[::-1]
			ppm = generate_pixel_data(frame_agg)
			
			img = Image.fromarray(ppm)
			#img.save('thumbnails/' + col_names[a] + '-frame' + str(f) + '.jpeg')
			#width, height = img.size
			#img = img.resize((width, height))

			print(calculate_score(ppm, 1))

			display(img)


def get_group(axes, aggs, agg_info, frames, mode):
	ppms = find_ppms(axes, aggs, agg_info, frames)
	scores = []
	for p in ppms:
		scores.append(calculate_score(p, mode))
	return ppms, scores