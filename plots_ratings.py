# plot players' ratings

import matplotlib.pyplot as plt
from pprint import pprint

from pipe import (
	Pipe,
	map as map_,
)
list_ = Pipe(list)



plt.rcParams.update({
	"font.size": 14,   # base font size (affects most text)
	# "axes.titlesize": 18,
	# "axes.labelsize": 16,
	# "xtick.labelsize": 12,
	# "ytick.labelsize": 12,
	# "legend.fontsize": 12
})



with open('./ratings_data.txt', 'r') as f:
	text = f.read()

epochs = text.split('\n\n')
# pprint(epochs)
# print(len(epochs))
# print('splited \\n\\n')
# input('press enter to continue...')

epochs = [epoch.split('\n') for epoch in epochs]
# pprint(epochs)
# print('splited \\n')
# input('press enter to continue...')

epochs = [[line for line in epoch if not line.startswith('//')] for epoch in epochs]
# pprint(epochs)
# print('removed //')
# input('press enter to continue...')

epochs = [[line.split('  ', 1)[0] for line in epoch] for epoch in epochs]
# pprint(epochs)
# print('splited "  "')
# input('press enter to continue...')

epochs = [[line.split('\t', 1) for line in epoch if line != ""] for epoch in epochs]
# pprint(epochs)
# print('splited "\\t"')
# input('press enter to continue...')

epochs = [[(float(rating[:-1]),name) for rating,name in epoch] for epoch in epochs]

epochs = [[(rating,name[3:]) for rating,name in epoch if name.startswith("NN ")] for epoch in epochs]

# pprint(epochs)
# for epoch in epochs:
# 	print(len(epoch))

epochs_n = len(epochs)
players = {name: [0.]*epochs_n for epoch in epochs for _rating,name in epoch}
# print(players)

best_players = []
for epoch in epochs:
	best_player = max(epoch, key=lambda rating_name: rating_name[0])
	best_players.append(best_player[1])
pprint(best_players)

for (epoch_i, epoch) in enumerate(epochs):
	for rating, name in epoch:
		players[name][epoch_i] = rating

# pprint(players)

for player, history in players.items():
	# plt.plot(history, label=player)
	alpha = 1.0 if player in best_players else 0.1
	plt.plot(history, alpha=alpha)

for epoch_i,epoch in enumerate(epochs):
	best_player = max(epoch, key=lambda rating_name: rating_name[0])
	plt.text(
		epoch_i,
		best_player[0],
		best_player[1],
		# fontsize=10,
		ha="left",
		va="bottom",
		rotation=90,
	)

plt.ylim(1000, 1450)
plt.xlim(0, epochs_n)

plt.xlabel('epoch number')
plt.ylabel('rating')
# plt.title('Player Ratings Over Time')
# plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.margins(left=.02, y=0)
plt.subplots_adjust(
	bottom=.05,
	left=.05,
	top=.98,
	right=.99,
)

plt.show()

