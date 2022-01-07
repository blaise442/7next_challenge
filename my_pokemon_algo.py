import requests
import pandas as pd
import pickle # serialization and deserilaization of our objects
import os.path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def add_pokemon(pokemon, pokemons):
	pokemon_name = pokemon['name']
	pokemon_type = pokemon['types'][0]['type']['name']
	l = []
	abilities = []
	for ability in pokemon['abilities']:
		abilities.append(ability['ability']['name'])
	moves = []
	for move in pokemon['moves']:
		moves.append(move['move']['name'])
	pokemons.append([pokemon_name, abilities, moves, pokemon_type])
	return pokemons

def predict(pokemon_):
	'''
	This function takes as input the json response for a pokemon through pokeapi and return the pokemon type as a string
	return: (string) pokemon type
	'''
	#Gets the count of total number of pokemons data available in the API
	count = requests.get("https://pokeapi.co/api/v2/pokemon/").json()['count']
	#Creating the pokemons dataframe
	pokemons = [['Name', 'Abilities', 'Moves', 'Type']]
	for i in range(1, count+1):
		pokemon = requests.get("https://pokeapi.co/api/v2/pokemon/{0}".format(i))
		try:
			pokemon = pokemon.json()
		except ValueError:
			continue
		pokemon_name = pokemon['name']
		pokemon_type = pokemon['types'][0]['type']['name']
		l = []
		abilities = []
		for ability in pokemon['abilities']:
			abilities.append(ability['ability']['name'])
		moves = []
		for move in pokemon['moves']:
			moves.append(move['move']['name'])
		pokemons.append([pokemon_name, abilities, moves, pokemon_type])
	pokemons = add_pokemon(pokemon_, pokemons)
	headers = pokemons.pop(0)
	df = pd.DataFrame(pokemons, columns=headers)
	#Label encoding the columns since string or list based datatypes can't be used for training the classifier
	le_name = LabelEncoder().fit(df['Name'].tolist())
	df['Name'] = le_name.transform(df['Name'].tolist())
	le_type = LabelEncoder().fit(df['Type'].tolist())
	df['Type'] = le_type.transform(df['Type'].tolist())
	flattened_abilities = [e for sublist in df['Abilities'].tolist() for e in sublist]
	le_abilties = LabelEncoder().fit(flattened_abilities)
	res_abilities = [list(le_abilties.transform(sublist)) for sublist in df['Abilities'].tolist()]
	df['Abilities'] = res_abilities
	flattened_moves = [e for sublist in df['Moves'].tolist() for e in sublist]
	le_moves = LabelEncoder().fit(flattened_moves)
	res_moves = [list(le_moves.transform(sublist)) for sublist in df['Moves'].tolist()]
	df['Moves'] = res_moves
	#Selecting the first element from list based values
	ab_list = []
	mv_list = []
	for i in range(len(df)):
	  if len(df.iloc[i]['Abilities']) > 0:
	    ab_list.append(df.iloc[i]['Abilities'][0])
	  else:
	    ab_list.append(0)
	  if len(df.iloc[i]['Moves']) > 0:
	    mv_list.append(df.iloc[i]['Moves'][0])
	  else:
	    mv_list.append(0)
	df['Abilities'] = ab_list
	df['Moves'] = mv_list
	# Putting feature variable to X
	X = df.drop('Type',axis=1)
	# Putting response variable to y
	y = df['Type']
	#Check if saved model is present
	if os.path.isfile('classifier_rf.sav'):
		classifier_rf = pickle.load(open(filename, 'rb'))
		return str(le_type.inverse_transform(classifier_rf.predict(X))[-1])
	#Initializing the random forest classifier
	classifier_rf = RandomForestClassifier(n_estimators=1500)
	#Training the random forest classifier
	classifier_rf.fit(X, y)
	filename = 'classifier_rf.sav'
	pickle.dump(classifier_rf, open(filename, 'wb'))
	return str(le_type.inverse_transform(classifier_rf.predict(X))[-1])