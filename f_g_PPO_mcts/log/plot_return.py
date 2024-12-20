import re
import matplotlib.pyplot as plt
import seaborn as sns

# read log file
log_file = 'training_log.log'

rounds = []
returns = []

pattern = re.compile(r'Round (\d+) return: (-?\d+)')

with open(log_file, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            round_number = int(match.group(1))
            return_value = int(match.group(2))
            rounds.append(round_number)
            returns.append(return_value)

# sort by round
# sorted_rounds, sorted_returns = zip(*sorted(zip(rounds, returns)))

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.lineplot(x=range(len(returns)), y=returns)
plt.title('Return Values per Round')
plt.xlabel('Round')
plt.ylabel('Return')
plt.show()