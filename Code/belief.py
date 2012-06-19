#!/usr/bin/env python2.5
# -*- coding: utf-8 *-*

import numpy as np
import scipy as sp
import monte.train
from scipy.sparse import lil_matrix, rand as sprand
from numpy.random import permutation
from monte.models.crf import ChainCrfLinear
from mingus.containers import Note, NoteContainer, Bar, Composition, Instrument, Track
from structures import Soprano, Alto, Tenor, Bass, NoteNode, NoteList, create_note_lists
from rules import *
from errors import *
from species import first_species

class belief (object):

    def __init__(self, composition, genre=None, fill_type="generate"):
        this._genre = genre # one of 'Bach', 'Jazz', 'Nursery'
        this._composition = composition
        this._fill_type = fill_type
        this._errors = []

    def train():
        for error in first_species(compostion):
            # print get_error_text(error)
            this._errors.append(error)

            rule = error[-1]
            if rule in written_rules:
                #print "Rule:", written_rules[rule]
                adjust_composition(rule)

        H = generate_H(errors,np.shape(errors))
        G = generatormatrix(H)

        if genre is None:
            this._genre = determine_genre(G)

    def test():
        compute_Belief_Prop(H)

        for u in range(len(B)):
            iterate_BP(200,u)

    def generate_H(m,n,d):
        ''' generate a parity check matrix H (m*n), density d (0<d<1) '''
        H = lil_matrix(m,n)
        H %= 2

        while not (all(sum(H,1)>=2) and all(sum(H,2)>=2)):
            H += abs(sprand(m,n,d)) > 0
            H %= 2

    def generatormatrix(H):
        ''' compute generator matrix from sparse parity matrix that we generated above '''

        m,n = np.shape(H)
        m,n = sorted([m,n], key=length)

        perm = permutation(1,n)

        for j in range(m):
            i = min(nonzero(H[j:m,j]))
            if not i: # then who?
                k = min(max(nonzero(H[j,:]),j))
                if not k:
                    print "Problem with matricies!"
                    raise
                temp   = H[:,j]
                H[:,j] = H[:,k]
                H[:,k] = temp
                temp    = perm[k]
                perm[k] = perm[j]
                perm[j] = temp

            i += j-1
            if (i != j):
                temp   = H[j,:]
                H[j,:] = H[i,:]
                H[i,:] = temp
            K = nonzero(H[:,j])
            K = K(nonzero(K!=j))
            if a.size(K):
                t1 = np.matrix(H[k,:]); # convert to full matrix
                for k in np.transpose(K):
                    t2 = np.matrix(H[k,:])
                    temp = xor(t1,t2)
                    H[k,:] = lil_matrix(temp)
        A = H[:,m+1:n]

        b, invperm = sorted(permutation(j)) #?
        G = [A, np.eye(n-m)]
        G = G[inverperm,:]

    def compute_Belief_Prop(H):
        ''' generate the matricies for P, S_ and q from H '''

        global B, P, S_, q, m, n

        m,n = np.shape(H)
        q = np.count_nonzero(H)
        P = lil_matrix(q,q, np.transpose(np.sum(H,2)) * np.sum(H,2))
        S = lil_matrix(q,q, (np.sum(H,1)-1) * np.transpose(np.sum(H,1)))

        k = 0
        for j in range(1,n):
            I = nonzero(H[:,j])
            for x in range(1,length(I)):
                for y in range(x+1,length(I)):
                    P[k+x,k+y] = 1
                    P[k+y,k+x] = 1
            k += length(I)

        k = 0
        for i in range(1,m):
            J = nonzero(H[i,:])
            for x in range(1,length(J)):
                for y in range(x+1,length(J)):
                    S_[k+x,k+y] = 1
                    S_[k+y,k+x] = 1

            k += length(J)

        B = lil_matrix(q,n,q)
        b = []
        for k in range(1,m):
            b = [nonzero(H[k,:])]
        B = lil_matrix(np.transpose([1,q]),np.transpose(b),np.ones(q,1),q,n)

    def compute_S(x):
        ''' compute the nonlinearity of the belief prop system
            S(x) is obtained by applying arctanh to x at i in S_ '''

        global S_, q

        y = np.ones(q,1)
        for i in (i,q):
            for j in nonzero(S_[i,:]):
                y[i] *= np.arctanh(x[j]/2)
        y = 2 * np.arctanh(y)

    def iterate_BP(iter_count, u):
        ''' iterate our belief prop over the notes list. The initial state is always 0'''

        global B, P, S_, q, n, m

        x1_k, x2_k, x1_k_1, x2_k_1 = np.zeros(q,1), np.zeros(q,1), np.zeros(q,1), np.zeros(q,1)

        y = np.zeros(n,iter_count+1)
        for i in range (1,iter_count):
            x1_k_1 = P * x2_k + B * u
            x2_k_1 = S_[x1_k]
            y[:,i] = np.transpose(B) * x2_k + u
            x1_k = x1_k_1
            x2_k = x2_k_1

        for node in composition.nodelist:
            node.belief = ((node.prev.belief * np.transpose(B)) + \
                            (node.next.belief * np.transpose(B))) / 2
            node.confidence = np.transpose(B) * x2_k + u



    def determine_genre():
        genrecrf = ChainCrfLinear(len(noteslist),len(voice))
        trainer = monte.train.Conjugategradients(genrecrf,5)
        genres = ["Jazz", "Bach", "Nursery"]
        for i in range(genres):
            trainer.step((len(noteslist), len(voice)), 0.01)
            cost = genrecrf.cost((inputs, outputs), 0.001)
            print("Cost: {0} ".genrecrf.cost((inputs, outputs), 0.001))

        return genres[0] if cost > 1 else genres[1]


