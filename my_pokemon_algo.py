def predict(pokemon):
	'''
	This function takes as input the json response for a pokemon through pokeapi and return the pokemon type as a string
	return: (string) pokemon type
	'''
	return pokemon['types'][0]['type']['name']