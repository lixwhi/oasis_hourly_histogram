import csv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timezone
from web3 import Web3
import json
import os
matplotlib.rc('xtick', labelsize=5)     
matplotlib.rc('ytick', labelsize=5)

gree = "#74e54708"
ree = "#e5475408"
gold = "#e5b64dff"
silver = '#a5a59dff'
grey3 = "#3d3d3dff"
ill_bid_color = '#59d4e899'
oll_bid_color = '#39bdc899'
ill_ask_color = '#fac0e199'
oll_ask_color = '#caa5f199'

CIRC_SCALAR = 10000
DAIMETER_MAX = 100000
DAIMETER_MIN = 250

XMIN_PLOT = 0.96
XMAX_PLOT = 1.04
YMIN_PLOT_DAI_REMAINING = 100
YMAX_PLOT_DAI_REMAINING = 100000
YMIN_PLOT_HIST = 0
YMAX_PLOT_HIST = 0.4
NUM_BINS = 25

MIN_MISPRICED_DAI = 2000
INNER_LIQUID_LINE = MIN_MISPRICED_DAI
OUTER_LIQUID_LINE = 20000

MIN_BLOCKS_MISPRICED = 40
MAX_DAIUSD_ERROR = 0.15

#ethusd_filename = "gemini_ETHUSD_2019_1min.csv"
eth2dai_data_dir = 'eth2dai_data'
ethusd_data_dir = 'ethusd_data'
#filename ='7214010-7218246_20190213.csv'
csv_outfile = 'DAIUSD_hourly_error_signal-04v2'
png_output_filename = "DAIUSD_hourly_error"


def main():
	hour_count = 0
	timestamp_out = []
	inner_price_error_out = []
	inner_std_out = []
	outer_price_error_out = []
	outer_std_out = []
	list_of_eth2dai = os.listdir(eth2dai_data_dir)
	list_of_ethusd = os.listdir(ethusd_data_dir)
	list_of_eth2dai.sort()
	num_hours_to_get = len(list_of_eth2dai) * 24
	print('getting {0}'.format(list_of_eth2dai))
	for v in range(0, len(list_of_eth2dai)):
		# block num
		x = np.loadtxt(open(eth2dai_data_dir + '/' + list_of_eth2dai[v], "rb"), delimiter=",", skiprows=1, usecols=0).astype(np.int32)
		# offer price
		y = np.loadtxt(open(eth2dai_data_dir + '/' + list_of_eth2dai[v], "rb"), delimiter=",", skiprows=1, usecols=1).astype(np.float64)
		# offer size in DAI
		s = np.loadtxt(open(eth2dai_data_dir + '/' + list_of_eth2dai[v], "rb"), delimiter=",", skiprows=1, usecols=2).astype(np.float64)
		# WETH/DAI bid = 1, WETH/DAI ask = -1
		boa = np.loadtxt(open(eth2dai_data_dir + '/' + list_of_eth2dai[v], "rb"), delimiter=",", skiprows=1, usecols=3).astype(int)
		# timestamp
		ts = np.loadtxt(open(eth2dai_data_dir + '/' + list_of_eth2dai[v], "rb"), delimiter=",", skiprows=1, usecols=4).astype(np.int32)
		# assign red to asks and green to bids
		c = np.empty(boa.size, dtype=np.dtype(('U10', 1)))
		c[boa == 1] = gree
		c[boa == -1] = ree



		# find the liquidity lines
		# this is the price that each level of DAI is available/for sale
		dilb = np.empty(x.size)
		dolb = np.empty(x.size)
		dila = np.empty(x.size)
		dola = np.empty(x.size)
		dilb, dolb, dila, dola = find_liquid_lines(dilb, dolb, dila, dola, x, y, s, boa)

		# go through and assign vals to ethusd price
		ethusd_timestamp = np.loadtxt(open(ethusd_data_dir + '/' + list_of_ethusd[0], "rb"), delimiter=",", skiprows=1, usecols=0)
		# reduce ethusd timestamp to match dex trade timestamps
		ethusd_timestamp = ethusd_timestamp / 1000
		ethusd_timestamp = ethusd_timestamp.astype(int)

		ethusd_high = np.loadtxt(open(ethusd_data_dir + '/' + list_of_ethusd[0], "rb"), delimiter=",", skiprows=1, usecols=4)
		ethusd_low = np.loadtxt(open(ethusd_data_dir + '/' + list_of_ethusd[0], "rb"), delimiter=",", skiprows=1, usecols=5)
		# use the midpoint of the high and low for that minute on gemini
		ethusd_mid = (ethusd_high + ethusd_low) / 2

		# Since I don't have ethusd data for every second, I must assume the ethusd price leads the
		# daiusd dex price. This is often the case because it takes a few blocks to confirm transactions 
		# after they are submitted. However, there is still information lost because the quality of the 
		# ethusd data. If I had ethusd pricing data down to the second, I would use it. For now,
		# I'm just going to round the dex prices down to the nearest minute. This will cause added
		# error to the daiusd error signal.
		# init arrays
		adjusted_dex_timestamp = np.empty(x.size)
		timestamp_hour = np.empty(x.size)
		timestamp_day = np.empty(x.size)
		timestamp_month = np.empty(x.size)
		ethusd_at_blocktime = np.empty(x.size)

		# round down blocktimes to nearest minute
		for i in range(0, x.size):
			dayt = datetime.fromtimestamp(ts[i])
			tstamp = datetime(year=dayt.year, month=dayt.month, day=dayt.day, hour=dayt.hour, minute=dayt.minute)
			tstamp.replace(tzinfo=timezone.utc)
			timestamp_hour[i] = tstamp.hour
			timestamp_day[i] = tstamp.day
			timestamp_month[i] = tstamp.month
			adjusted_dex_timestamp[i] = tstamp.timestamp()
		adjusted_dex_timestamp = adjusted_dex_timestamp.astype(int)
		timestamp_hour = timestamp_hour.astype(int)
		timestamp_day = timestamp_day.astype(int)
		timestamp_month = timestamp_month.astype(int)


		for i in range(0, adjusted_dex_timestamp.size):
			indx = np.where(ethusd_timestamp == adjusted_dex_timestamp[i])
			#print('{0} vs {1}'.format(ethusd_timestamp, adjusted_dex_timestamp[i]))
			if(indx[0].size != 0):
				ethusd_at_blocktime[i] = ethusd_mid[indx[0][0]]

		daiusd_price = ethusd_at_blocktime / y
		#save_debug4(ts, adjusted_dex_timestamp, ethusd_at_blocktime, daiusd_price)
		# scaling stuff
		scaled_s = np.copy(s)
		scaled_s[scaled_s > DAIMETER_MAX] = DAIMETER_MAX
		scaled_s[scaled_s < DAIMETER_MIN] = DAIMETER_MIN
		scaled_s = scaled_s / DAIMETER_MIN
		scaled_s = np.square(scaled_s)
		# now 0-1
		scaled_s = scaled_s / ((DAIMETER_MAX / DAIMETER_MIN) ** 2)
		scaled_s = scaled_s * CIRC_SCALAR
		list_of_daiusd_prices = []
		list_of_scaled_sizes = []
		list_of_actual_sizes = []
		list_of_colors = []
		list_of_bid_inner = []
		list_of_bid_inner_std = []
		list_of_bid_outer = []
		list_of_bid_outer_std = []
		list_of_ask_inner = []
		list_of_ask_inner_std = []
		list_of_ask_outer = []
		list_of_ask_outer_std = []
		prev_hour= timestamp_hour[0]
		first_hour = timestamp_hour[0]
		first_hour_flag = True

		fig = plt.figure()

		
		blocks_mispriced_count_bids = 0
		blocks_mispriced_count_asks = 0
		perror = 0
		list_perror = []
		prev_bl_bid = x[0]
		prev_bl_ask = x[0]
		inner_error_sig = 0
		outer_error_sig = 0


		for i in range(0, daiusd_price.size):
			list_of_daiusd_prices.append(daiusd_price[i])
			list_of_scaled_sizes.append(scaled_s[i])
			list_of_actual_sizes.append(s[i])
			list_of_colors.append(c[i])


			#this needs to add the inner and out liquid lines together
			# then build a histogram and find the median
			# and plt overlap the histograms below with a different y axis
			# then add two plots that increment each frame
			# one for inner error and another for outer 
			list_of_bid_inner.append(ethusd_at_blocktime[i] / dilb[i])
			list_of_bid_outer.append(ethusd_at_blocktime[i] / dolb[i])
			list_of_ask_inner.append(ethusd_at_blocktime[i] / dila[i])
			list_of_ask_outer.append(ethusd_at_blocktime[i] / dola[i])

			if ((len(list_of_bid_inner) <= 7000) and (first_hour_flag)):
				first_hour_flag = True
			elif(first_hour_flag):
				first_hour_flag = False
				prev_hour = timestamp_hour[i]
			


			# save plot stuff once per hour
			if ((timestamp_hour[i] != prev_hour) and (not first_hour_flag)):
				bi = np.asarray(list_of_bid_inner)
				bo = np.asarray(list_of_bid_outer)
				ai = np.asarray(list_of_ask_inner)
				ao = np.asarray(list_of_ask_outer)
				bi_med = round(np.median(bi), 4)
				bo_med = round(np.median(bo), 4)
				ai_med = round(np.median(ai), 4)
				ao_med = round(np.median(ao), 4)
				bi_std = round(np.std(bi), 6)
				bo_std = round(np.std(bo), 6)
				ai_std = round(np.std(ai), 6)
				ao_std = round(np.std(ao), 6)
				inner_error_sig = 1
				outer_error_sig = 1
				inner_error_std = 0
				outer_error_std = 0
				if(bi_med < 1):
					inner_error_sig = bi_med
					inner_error_std = bi_std
				if(bo_med < 1):
					outer_error_sig = bo_med
					outer_error_std = bo_std
				if(ai_med > 1):
					inner_error_sig = ai_med
					inner_error_std = ai_std
				if(ao_med > 1):
					outer_error_sig = ao_med
					outer_error_std = ao_std

				
				daiprice = np.asarray(list_of_daiusd_prices)
				scaled_size = np.asarray(list_of_scaled_sizes)
				actual_size = np.asarray(list_of_actual_sizes)
				col = np.asarray(list_of_colors)

				timestamp_out.append(ts[i-1])
				inner_price_error_out.append(inner_error_sig - 1)
				inner_std_out.append(inner_error_std)
				outer_price_error_out.append(outer_error_sig - 1)
				outer_std_out.append(outer_error_std)

				
				plot_and_save(inner_error_sig, outer_error_sig, daiprice, actual_size, col, hour_count, ts[i], bi, bo, ai, ao, inner_price_error_out, outer_price_error_out, inner_std_out, outer_std_out, num_hours_to_get)

				prev_hour = timestamp_hour[i]
				hour_count += 1
				list_of_daiusd_prices = []
				list_of_scaled_sizes = []
				list_of_actual_sizes = []
				list_of_colors = []
				list_of_bid_inner = []
				list_of_bid_inner_std = []
				list_of_bid_outer = []
				list_of_bid_outer_std = []
				list_of_ask_inner = []
				list_of_ask_inner_std = []
				list_of_ask_outer = []
				list_of_ask_outer_std = []

				perror = 0
				list_perror = []

		# make sure to save the last hour
		last_time = datetime.fromtimestamp(ts[ts.size - 1])
		if(last_time.minute > 30):
			bi = np.asarray(list_of_bid_inner)
			bo = np.asarray(list_of_bid_outer)
			ai = np.asarray(list_of_ask_inner)
			ao = np.asarray(list_of_ask_outer)
			bi_med = round(np.median(bi), 4)
			bo_med = round(np.median(bo), 4)
			ai_med = round(np.median(ai), 4)
			ao_med = round(np.median(ao), 4)
			bi_std = round(np.std(bi), 6)
			bo_std = round(np.std(bo), 6)
			ai_std = round(np.std(ai), 6)
			ao_std = round(np.std(ao), 6)
			inner_error_sig = 1
			outer_error_sig = 1
			inner_error_std = 0
			outer_error_std = 0
			if(bi_med < 1):
				inner_error_sig = bi_med
				inner_error_std = bi_std
			if(bo_med < 1):
				outer_error_sig = bo_med
				outer_error_std = bo_std
			if(ai_med > 1):
				inner_error_sig = ai_med
				inner_error_std = ai_std
			if(ao_med > 1):
				outer_error_sig = ao_med
				outer_error_std = ao_std

			daiprice = np.asarray(list_of_daiusd_prices)
			scaled_size = np.asarray(list_of_scaled_sizes)
			actual_size = np.asarray(list_of_actual_sizes)
			col = np.asarray(list_of_colors)
			timestamp_out.append(ts[i-1])
			inner_price_error_out.append(inner_error_sig - 1)
			inner_std_out.append(inner_error_std)
			outer_price_error_out.append(outer_error_sig - 1)
			outer_std_out.append(outer_error_std)
			plot_and_save(inner_error_sig, outer_error_sig, daiprice, actual_size, col, hour_count, ts[ts.size - 1], bi, bo, ai, ao, inner_price_error_out, outer_price_error_out, inner_std_out, outer_std_out, num_hours_to_get)
			hour_count += 1
			
			

	save_error_signal(timestamp_out, inner_price_error_out, outer_price_error_out, inner_std_out, outer_std_out)

def save_error_signal (timestamp_out, in_error_out, out_error_out, in_std_out, out_std_out):
	first_hour = datetime.fromtimestamp(timestamp_out[0])
	last_hour = datetime.fromtimestamp(timestamp_out[len(timestamp_out) - 1])
	with open('{0}-{1}_{2}.csv'.format(first_hour.strftime("%Y%m%d%H"), last_hour.strftime("%Y%m%d%H"), csv_outfile), 'w+') as csvF:
		writer = csv.writer(csvF)
		head = ['timestamp', '{0}_depth_of_market'.format(INNER_LIQUID_LINE), '{0}_depth_of_market_standard_dev'.format(INNER_LIQUID_LINE),'{0}_depth_of_market'.format(OUTER_LIQUID_LINE), '{0}_depth_of_market_standard_dev'.format(OUTER_LIQUID_LINE)]
		writer.writerow(head)
		for i in range(0, len(timestamp_out)):
			row = [int(timestamp_out[i]), in_error_out[i], in_std_out[i], out_error_out[i], out_std_out[i]]
			writer.writerow(row)
	csvF.close()



def plot_and_save(inner_err, outer_err, daiprice, actual_size, col, hour_count, times, bi, bo, ai, ao, ipe, ope, inner_std_out, outer_std_out, num_hours):
	print('ploting {0}\n'.format(hour_count))
	
	fig = plt.figure()
	ax1 = plt.subplot2grid((4,3), (0,0), rowspan=2, colspan=3)
	ax2 = plt.subplot2grid((4,3), (2,0), rowspan=1, colspan=3)
	ax3 = plt.subplot2grid((4,3), (3,0), rowspan=1, colspan=3)

	#bi_hist, bi_edge = np.histogram(bi, bins=NUM_BINS, range=(XMIN_PLOT, XMAX_PLOT))
	#bo_hist, bo_edge = np.histogram(bo, bins=NUM_BINS, range=(XMIN_PLOT, XMAX_PLOT))
	#ai_hist, ai_edge = np.histogram(ai, bins=NUM_BINS, range=(XMIN_PLOT, XMAX_PLOT))
	#ao_hist, ao_edge = np.histogram(ao, bins=NUM_BINS, range=(XMIN_PLOT, XMAX_PLOT))

	ax1.set_xlim(XMIN_PLOT, XMAX_PLOT)
	ax1.set_xlabel('DAI/USD', fontsize=5)
	ax1.axvline(1, linewidth=0.75)
	ax1.axhline(INNER_LIQUID_LINE, linewidth=0.25)
	ax1.axhline(OUTER_LIQUID_LINE, linewidth=0.25)
	#ax1.annotate('min size\nthreshold', (XMAX_PLOT, MIN_MISPRICED_DAI), xytext=(XMAX_PLOT, MIN_MISPRICED_DAI), xycoords='data',textcoords='data', arrowprops=None, fontsize=6)
	ax1.annotate('    -   {0} (median) market depth = {1}'.format(INNER_LIQUID_LINE, round(inner_err - 1, 4)), (XMIN_PLOT, YMAX_PLOT_DAI_REMAINING / 2.2), xytext=(XMIN_PLOT, YMAX_PLOT_DAI_REMAINING / 2.2), xycoords='data',textcoords='data', arrowprops=None, fontsize=5)
	ax1.annotate('    - {0} (median) market depth = {1}'.format(OUTER_LIQUID_LINE, round(outer_err - 1, 4)), (XMIN_PLOT, YMAX_PLOT_DAI_REMAINING / 5), xytext=(XMIN_PLOT, YMAX_PLOT_DAI_REMAINING / 5), xycoords='data',textcoords='data', arrowprops=None, fontsize=5)
	#ax1.annotate('median hourly {0} depth - '.format(INNER_LIQUID_LINE), (XMAX_PLOT * 0.7, YMAX_PLOT_DAI_REMAINING / 1.5), xytext=(XMAX_PLOT * 0.7, YMAX_PLOT_DAI_REMAINING / 1.5), xycoords='data',textcoords='data', arrowprops=None, fontsize=8)
	#ax1.annotate('median hourly {0} depth - '.format(OUTER_LIQUID_LINE), (XMAX_PLOT * 0.7, YMAX_PLOT_DAI_REMAINING / 2.2), xytext=(XMAX_PLOT * 0.7, YMAX_PLOT_DAI_REMAINING / 2.2), xycoords='data',textcoords='data', arrowprops=None, fontsize=8)
	ax1.annotate('X'.format(INNER_LIQUID_LINE), (XMIN_PLOT, YMAX_PLOT_DAI_REMAINING / 2.2), xytext=(XMIN_PLOT, YMAX_PLOT_DAI_REMAINING / 2.2), xycoords='data',textcoords='data', arrowprops=None, fontsize=8, color=gold)
	ax1.annotate('X'.format(OUTER_LIQUID_LINE), (XMIN_PLOT, YMAX_PLOT_DAI_REMAINING / 5), xytext=(XMIN_PLOT, YMAX_PLOT_DAI_REMAINING / 5), xycoords='data',textcoords='data', arrowprops=None, fontsize=8, color=grey3)

	hourr = datetime.fromtimestamp(times)
	ax1.annotate('hour: {0}'.format(hourr.strftime("%Y%m%d-{0}".format(hour_count))), (XMAX_PLOT - 0.015, YMAX_PLOT_DAI_REMAINING / 2.2), xytext=(XMAX_PLOT - 0.015, YMAX_PLOT_DAI_REMAINING / 2.2), xycoords='data',textcoords='data', arrowprops=None, fontsize=5)
	ax1.set_ylim(YMIN_PLOT_DAI_REMAINING, YMAX_PLOT_DAI_REMAINING)
	ax1.set_ylabel('DAI remaining', fontsize=5)
	ax1.set_yscale("log")
	ax1.set_title("Oasis orderbook converted to DAI/USD")
	ax1.vlines(daiprice, [0], actual_size, colors=col, linewidth=1.2)
	ax1.plot(inner_err, INNER_LIQUID_LINE, marker='x', color=gold, markeredgecolor=gold, markerfacecolor=gold, markersize=12)
	ax1.plot(outer_err, OUTER_LIQUID_LINE, marker='x', color=grey3, markeredgecolor=grey3, markerfacecolor=grey3, markersize=12)
	
	#plt.setp(lines, pickradius=0.2)
	#ax2 = plt.subplot(312)
	ax2.set_xlim(XMIN_PLOT, XMAX_PLOT)
	ax2.set_ylabel('bid/ask depth hourly density', fontsize=5)
	ax2.yaxis.set_label_position("right")
	ax2.axvline(np.median(bi), linewidth=0.25, color=gold)
	ax2.axvline(np.median(bo), linewidth=0.25, color=grey3)
	ax2.axvline(np.median(ai), linewidth=0.25, color=gold)
	ax2.axvline(np.median(ao), linewidth=0.25, color=grey3)
	ax2.annotate('hourly WETH/DAI {0} bid depth'.format(INNER_LIQUID_LINE), (XMIN_PLOT, 0.35), xytext=(XMIN_PLOT, 0.35), xycoords='data',textcoords='data', arrowprops=None, fontsize=8, color=ill_bid_color)
	ax2.annotate('hourly WETH/DAI {0} bid depth'.format(OUTER_LIQUID_LINE), (XMIN_PLOT, 0.3), xytext=(XMIN_PLOT, 0.3), xycoords='data',textcoords='data', arrowprops=None, fontsize=8, color=oll_bid_color)
	ax2.annotate('hourly WETH/DAI {0} ask depth'.format(INNER_LIQUID_LINE), (XMIN_PLOT, 0.25), xytext=(XMIN_PLOT, 0.25), xycoords='data',textcoords='data', arrowprops=None, fontsize=8, color=ill_ask_color)
	ax2.annotate('hourly WETH/DAI {0} ask depth'.format(OUTER_LIQUID_LINE), (XMIN_PLOT, 0.2), xytext=(XMIN_PLOT, 0.2), xycoords='data',textcoords='data', arrowprops=None, fontsize=8, color=oll_ask_color)


	ax2.set_ylim(YMIN_PLOT_HIST, YMAX_PLOT_HIST)
	if (bi.size != 0):
		weights_bi = np.ones_like(bi)/float(len(bi))
		ax2.hist(bi, bins=NUM_BINS, weights=weights_bi, color=ill_bid_color)
	if (bo.size != 0):
		weights_bo = np.ones_like(bo)/float(len(bo))
		ax2.hist(bo, bins=NUM_BINS, weights=weights_bo, color=oll_bid_color)
	if (ai.size != 0):
		weights_ai = np.ones_like(ai)/float(len(ai))
		ax2.hist(ai, bins=NUM_BINS, weights=weights_ai, color=ill_ask_color)
	if (ao.size != 0):
		weights_ao = np.ones_like(ao)/float(len(ao))
		ax2.hist(ao, bins=NUM_BINS, weights=weights_ao, color=oll_ask_color)
	
	#ax3 = plt.subplot(313)
	inpe = np.asarray(ipe)
	instd = np.asarray(inner_std_out)
	outpe = np.asarray(ope)
	outstd = np.asarray(outer_std_out)
	ax3.set_xlim(0, num_hours)
	ax3.set_ylim(-0.04, 0.04)
	ax3.set_ylabel('DAI/USD error signal', fontsize=5)
	ax3.set_xlabel('hours', fontsize=5)
	ax3.axhline(0, linewidth=0.25)
	#ax3.plot(inpe, color=gold)
	#ax3.plot(outpe, color=grey3)
	day0 = np.arange(0, inpe.size)
	ax3.errorbar(day0, inpe, yerr=instd, fmt='o', markersize=0.7, color=gold, capsize=0.8, elinewidth=0.5)
	ax3.errorbar(day0, outpe, yerr=outstd, fmt='o', markersize=0.7, color=grey3, capsize=0.8, elinewidth=0.5)

	#fig.tight_layout()
	fig.savefig(png_output_filename + str(hour_count) + '.png', dpi=250)
	fig.clf()
	plt.clf()
	plt.close()


def find_liquid_lines(dilb, dolb, dila, dola, x, y, s, boa):
	llbids = []
	llasks = []
	prev_bnum = x[0]
	bid_stack = []
	ask_stack = []
	for i in range(0, x.size):
		if(prev_bnum == x[i]):
			if(boa[i] == 1):
				bid_stack.append((y[i], s[i]))
				
			elif(boa[i] == -1):
				ask_stack.append((y[i], s[i]))		
		else:
			# sort bids by price
			bsort = sorted(bid_stack, key=lambda q: q[0])
			bsort.reverse()
			###print(bsort)
			# sort asks by price
			asort = sorted(ask_stack, key=lambda q: q[0])
			running_dai = 0
			ill = 0
			oll = 0
			# find liquid lines for bids
			for k in bsort:
				running_dai += k[1]
				###print(k[1])
				if((running_dai >= INNER_LIQUID_LINE) and (ill == 0)):
					ill = k[0]
					
				if((running_dai >= OUTER_LIQUID_LINE) and (oll == 0)):
					oll = k[0]
					break
			# check to make sure there is something here
			if((oll == 0) or (ill == 0)):
				ill = y[i] * 0.9999
				oll = y[i] * 0.9998
			tip = (prev_bnum, ill, oll)
			llbids.append(tip)

			# find liquid lines for asks
			running_dai = 0
			ill = 0
			oll = 0
			for k in asort:
				running_dai += k[1]
				if((running_dai >= INNER_LIQUID_LINE) and (ill == 0)):
					ill = k[0]
				if((running_dai >= OUTER_LIQUID_LINE) and (oll == 0)):
					oll = k[0]
					break
			# check to make sure there is something here
			if((oll == 0) or (ill == 0)):
				ill = y[i] * 1.0001
				oll = y[i] * 1.0002
			tip = (prev_bnum, ill, oll)
			llasks.append(tip)

			prev_bnum = x[i]
			# reset stacks and add new offer
			bid_stack = []
			ask_stack = []
			if(boa[i] == 1):
				bid_stack.append((y[i], s[i]))
			elif(boa[i] == -1):
				ask_stack.append((y[i], s[i]))

	# go through and assign vals to dil, dol
	ll_indx = 0
	bnum = x[0]
	ill_bid = llbids[0][1]
	oll_bid = llbids[0][2]
	ill_ask = llasks[0][1]
	oll_ask = llasks[0][2]
	for i in range(0, x.size):
		if(bnum != x[i]):
			bnum = x[i]
			ill_bid = llbids[ll_indx][1]
			oll_bid = llbids[ll_indx][2]
			ill_ask = llasks[ll_indx][1]
			oll_ask = llasks[ll_indx][2]
			ll_indx += 1
		
		dilb[i] = ill_bid
		dolb[i] = oll_bid
		dila[i] = ill_ask
		dola[i] = oll_ask

	#save_debug(dilb, dolb, dila, dola)

	return dilb, dolb, dila, dola


def save_debug4(dilb, dolb, dila, dola):
	with open('debu4.csv', 'w+') as csvF2:
		writer = csv.writer(csvF2)

		for i in range(0, dilb.size):
			row = [dilb[i], dolb[i], dila[i], dola[i]]
			writer.writerow(row)
	csvF2.close()

def save_debug1(dilb):
	with open('debu1.csv', 'w+') as csvF2:
		writer = csv.writer(csvF2)

		for i in range(0, dilb.size):
			row = [dilb[i]]
			writer.writerow(row)
	csvF2.close()
def save_debug2(dilb, dolb):
	with open('debu2.csv', 'w+') as csvF2:
		writer = csv.writer(csvF2)

		for i in range(0, dilb.size):
			row = [dilb[i], dolb[i]]
			writer.writerow(row)
	csvF2.close()
# ll_indx = 0
# bnum = x[0]
# ill_bid = llbids[0][1]
# oll_bid = llbids[0][2]
# ill_ask = llasks[0][1]
# oll_ask = llasks[0][2]
# for i in range(0, x.size):
# 	if(bnum != x[i]):
# 		bnum = x[i]
# 		ill_bid = llbids[ll_indx][1]
# 		oll_bid = llbids[ll_indx][2]
# 		ill_ask = llasks[ll_indx][1]
# 		oll_ask = llasks[ll_indx][2]
# 		ll_indx += 1
# 	dilb[i] = ill_bid
# 	dolb[i] = oll_bid
# 	dila[i] = ill_ask
# 	dola[i] = oll_ask




# fig = plt.figure()
# ax1 = plt.axes()
# ax1.set_ylim(120, 128)
# #plt.scatter(x, y, s=s, c=c, marker="|")
# plt.scatter(x, dilb, c=ill_bid_color, marker=',')
# plt.scatter(x, dolb, c=oll_bid_color, marker='o')
# plt.scatter(x, dila, c=ill_ask_color, marker=',')
# plt.scatter(x, dola, c=oll_ask_color, marker='o')
# plt.title('Oasis book ' + outfile)
# #plt.savefig(outfile, dpi=2000)
# plt.show()


if __name__ == "__main__":
	main()