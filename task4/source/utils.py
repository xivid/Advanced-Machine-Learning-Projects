import pandas as pd

def save_solution(csv_file,prob_positive_class):
	with open(csv_file, 'w') as csv:
		df = pd.DataFrame.from_dict({'id':range(len(prob_positive_class)),'y': prob_positive_class})
		df.to_csv(csv,index = False)