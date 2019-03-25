import torch
from config import config
from torch.nn import functional as F

_config = config()


def evaluate(golden_list, predict_list):
	golden_counter = 0
	predict_counter = 0
	counter = 0

	for i in golden_list:
		for j in i:
			if j[0] == 'B':
				golden_counter += 1

	for i in predict_list:
		for j in i:
			if j[0] == 'B':
				predict_counter += 1

	position = []
	predict_position = []
	for i in range(len(golden_list)):
		for j in range(len(golden_list[i])):
			if golden_list[i][j] != 'B-TAR' and golden_list[i][j] != 'B-HYP':
				#print(golden_list[i][j])
				continue
			if j == len(golden_list[i]) - 1:
				if golden_list[i][j] == 'B-HYP':
					position.append((i,j,j+1,'B-HYP'))
				elif golden_list[i][j] == 'B-TAR':
					position.append((i,j,j+1, 'B-TAR'))
				#position.append((i, j, j+1))
			else:
				for k in range(j+1,len(golden_list[i])):
					if j+1 == len(golden_list[i])-1:
						if golden_list[i][j] == 'B-HYP' and golden_list[i][j+1] == 'I-HYP':
							position.append((i,j,j+2,'B-HYP'))
						elif golden_list[i][j] == 'B-TAR' and golden_list[i][j+1] == 'I-TAR':
							position.append((i,j,j+2, 'B-TAR'))	
						else:
							if golden_list[i][j] == 'B-HYP':
								position.append((i,j,k,'B-HYP'))
							elif golden_list[i][j] == 'B-TAR':
								position.append((i,j,k, 'B-TAR'))	
					else:
						if golden_list[i][k] == 'O' or golden_list[i][k][0] == 'B' or k == len(golden_list[i]) - 1:
							if golden_list[i][j] == 'B-HYP':
								position.append((i,j,k,'B-HYP'))
							elif golden_list[i][j] == 'B-TAR':
								position.append((i,j,k, 'B-TAR'))
							#position.append((i, j, k))
							break

	for i in range(len(predict_list)):
		for j in range(len(predict_list[i])):
			if predict_list[i][j] != 'B-TAR' and predict_list[i][j] != 'B-HYP':
				continue
			if j == len(predict_list[i]) - 1:
				if predict_list[i][j] == 'B-HYP':
					predict_position.append((i,j,j+1,'B-HYP'))
				elif predict_list[i][j] == 'B-TAR':
					predict_position.append((i,j,j+1, 'B-TAR'))
				#predict_position.append((i, j, j+1))
			else:
				for k in range(j+1,len(predict_list[i])):
					if j+1 == len(predict_list[i])-1:
						if predict_list[i][j] == 'B-HYP' and predict_list[i][j+1] == 'I-HYP':
							predict_position.append((i,j,j+2,'B-HYP'))
						elif predict_list[i][j] == 'B-TAR' and predict_list[i][j+1] == 'I-TAR':
							predict_position.append((i,j,j+2, 'B-TAR'))		
						else:
							if predict_list[i][j] == 'B-HYP':
								predict_position.append((i,j,k,'B-HYP'))
							elif predict_list[i][j] == 'B-TAR':
								predict_position.append((i,j,k, 'B-TAR'))											
					else:
						if predict_list[i][k] == 'O' or predict_list[i][k][0] == 'B' or k == len(golden_list[i]) - 1:
							if predict_list[i][j] == 'B-HYP':
								predict_position.append((i,j,k,'B-HYP'))
							elif predict_list[i][j] == 'B-TAR':
								predict_position.append((i,j,k, 'B-TAR'))						
							#predict_position.append((i, j, k))
							break

	counter = len([i for i in position if i in predict_position])
	precision = float()
	recall = float()
	score = float()
	if len(predict_position)==0 and len(position)==0 and counter == 0:
		score=1.000
		#print(score)
		return score
	else:
		if len(predict_position)==0 or len(position)==0:
			score=0.000
			#print(score)
			return score

		else:
			precision = counter / len(predict_position)
			recall = counter / len(position)
			if precision + recall == 0 or counter == 0:
				score = 0.000
				#print(score)
				return score
			else:
				score = (2 * ((precision*recall)/(precision+recall)))
				#sprint(score)
				return score


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused()
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = (forgetgate * cx) + (torch.ones_like(forgetgate)-forgetgate) * cellgate
    hy = outgate * F.tanh(cy)

    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    pass;


