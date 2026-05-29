# plot players' ratings

from sys import argv as cli_args
from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



plt.rcParams.update({
	"font.size": 14,   # base font size (affects most text)
	# "axes.titlesize": 18,
	# "axes.labelsize": 16,
	# "xtick.labelsize": 12,
	# "ytick.labelsize": 12,
	# "legend.fontsize": 12
})

FILENAME = cli_args[1]

fig, ax = plt.subplots()



def main():
	update_plot() # initial plot

	ani = FuncAnimation(
		fig,
		update_plot,
		interval=60_000, # 1 minute
	)

	plt.show()



def update_plot(frame=None):
	text = read_data(FILENAME)

	lines = text.splitlines()
	lines = [line for line in lines if line == "" or (line[0].isdigit() and ":\t" in line)]

	collapsed = []
	previous_empty = False
	for line in lines:
		is_empty = (line == "")
		if is_empty:
			if not previous_empty:
				collapsed.append(line)
		else:
			collapsed.append(line)
		previous_empty = is_empty
	lines = collapsed

	text = '\n'.join(lines)

	epochs = text.split("\n\n")
	# pprint(epochs)
	# print(len(epochs))
	# print("splited \\n\\n")
	# input("press enter to continue...")

	epochs = [epoch.split("\n") for epoch in epochs]
	# pprint(epochs)
	# print("splited \\n")
	# input("press enter to continue...")

	epochs = [[line for line in epoch if not line.startswith("//")] for epoch in epochs]
	# pprint(epochs)
	# print("removed //")
	# input("press enter to continue...")

	epochs = [[line.split("  ", 1)[0] for line in epoch] for epoch in epochs]
	# pprint(epochs)
	# print("splited "  "")
	# input("press enter to continue...")

	epochs = [[line.split("\t", 1) for line in epoch if line != ""] for epoch in epochs]
	# pprint(epochs)
	# print("splited "\\t"")
	# input("press enter to continue...")

	epochs = [[(float(rating[:-1]),name) for rating,name in epoch] for epoch in epochs]

	epochs = [[(rating,name[3:]) for rating,name in epoch if name.startswith("NN ")] for epoch in epochs]

	# pprint(epochs)
	# for epoch in epochs:
	# 	print(len(epoch))

	epochs_n = len(epochs)

	players = {name: [0.]*epochs_n for epoch in epochs for _rating,name in epoch}
	# print(players)
	for (epoch_i, epoch) in enumerate(epochs):
		for rating, name in epoch:
			players[name][epoch_i] = rating
	# pprint(players)

	best_players = []
	for epoch in epochs:
		best_player = max(epoch, key=lambda rating_name: rating_name[0])
		best_players.append(best_player[1])
	# pprint(best_players)

	def str_to_int(s: str) -> int:
		res = 0
		for c in s:
			res *= 256
			res += ord(c)
		# res = (res*100000000103 + res*1381 + res*27 + 17) ^ (res-193) + res//27 + res//1031 + res//100000000019
		res = res*100000000103 + res*1381 + res*27 + 17 + res//27 + res//1031 + res//100000000019
		# res += res % 8361103913
		res += res % 8361103937
		# res ^= ?
		return res

	def lerp(a: float, b: float, t: float):
		return (1-t)*a + t*b

	def skew_colors_1(r: float, g: float, b: float, k: float) -> tuple[float,float,float]:
		if r > g and r > b:
			r = lerp(r, 1, k)
			g = lerp(g, 0, k)
			b = lerp(b, 0, k)
		elif g > r and g > b:
			r = lerp(r, 0, k)
			g = lerp(g, 1, k)
			b = lerp(b, 0, k)
		elif b > r and b > g:
			r = lerp(r, 0, k)
			g = lerp(g, 0, k)
			b = lerp(b, 1, k)
		# else:
		# 	pass
		return r, g, b

	def skew_colors_2(r: float, g: float, b: float, k: float) -> tuple[float,float,float]:
		r = lerp(r, int(r > 0.5), k)
		g = lerp(g, int(g > 0.5), k)
		b = lerp(b, int(b > 0.5), k)
		return r, g, b

	def mod_frac(x: int, n: int) -> float:
		return (x % n) / n

	colormap = plt.colormaps.get_cmap("hsv") # hsv, turbo, jet
	def str_to_color(s: str) -> tuple[float,float,float] | tuple[float,float,float,float]:
		# return colormap(mod_frac(str_to_int(s), 2**16))
		n = str_to_int(s)
		n = n // 13
		r = (n % 256) / 256
		g = ((n >> 8) % 256) / 256
		b = ((n >> 16) % 256) / 256
		# return skew_colors_2(r, g, b, 0.3)
		# r2,g2,b2,_ = colormap(mod_frac(str_to_int(s), 189043))
		r2,g2,b2,_ = colormap(mod_frac(str_to_int(s), 18904309))
		# r2,g2,b2,_ = colormap(mod_frac(str_to_int(s), 14904209))
		# r2,g2,b2,_ = colormap(mod_frac(str_to_int(s), 1490429))
		# r2,g2,b2,_ = colormap(mod_frac(str_to_int(s), 3188791))
		# r2,g2,b2,_ = colormap(mod_frac(str_to_int(s), 251))
		r = lerp(r, r2, 0.8)
		g = lerp(g, g2, 0.8)
		b = lerp(b, b2, 0.8)
		return (r, g, b)

	# for name in best_players:
	# 	print(hash(name))
	# 	print(str_to_int(name))
	# 	print()

	players_colors = { name: str_to_color(name) for name in players }
	# players_colors = {}
	# for name in players:
	# 	if name in best_players:
	# 		players_colors[name] = ???
	# 	else:
	# 		players_colors[name] = str_to_color(name)

	ax.clear()

	for name, history in players.items():
		alpha = 1.0 if name in best_players else 0.1
		ax.plot(history, alpha=alpha, color=players_colors[name])

	for epoch_i,epoch in enumerate(epochs):
		rating, name = max(epoch, key=lambda rating_name: rating_name[0])
		ax.scatter(epoch_i, rating, s=30, color=players_colors[name])
		ax.annotate(
			name,
			(epoch_i, rating),
			xytext=(0, 15),
			textcoords="offset points",
			rotation=90,
			ha="center",
		)

	ax.set_xlim(0, epochs_n)

	max_rating = max(rating for epoch in epochs for rating,_name in epoch)
	ax.set_ylim(1000, 1000 + (max_rating-1000) * 1.3)

	ax.set_xlabel("epoch number")
	ax.set_ylabel("rating")
	# ax.title("Player Ratings Over Time")
	# ax.legend()
	ax.grid(True)
	# fig.tight_layout()
	# ax.margins(left=.02, y=0)
	fig.subplots_adjust(
		bottom=.06,
		left=.05,
		top=.98,
		right=.99,
	)

	# plt.show()





def read_data(filename: str) -> str:
	with open(filename, "r") as f:
		text = f.read()
	return text





if __name__ == "__main__":
	main()

