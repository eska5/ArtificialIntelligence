from itertools import product
import numpy as np
import matplotlib.pyplot as plt

F = 0
P = 1

pW = np.zeros(2)
pW[P] = 0.001
pW[F] = 0.999

pU = np.zeros(2)
pU[P] = 0.002
pU[F] = 0.998

pA = np.zeros((2,2,2))
pA[P,P,P]= 0.95
pA[P,P,F]= 0.05
pA[P,F,P]= 0.94
pA[P,F,F]= 0.06
pA[F,P,P]= 0.29
pA[F,P,F]= 0.71
pA[F,F,P]= 0.001
pA[F,F,F]= 0.999

pSA = np.zeros((2,2))
pSA[P,P] = 0.9
pSA[P,F] = 0.1
pSA[F,P] = 0.05
pSA[F,F] = 0.95

pBA = np.zeros((2,2))
pBA[P,P] = 0.7
pBA[P,F] = 0.3
pBA[F,P] = 0.01
pBA[F,F] = 0.99

p = np.zeros((2, 2, 2, 2, 2))
for W in [P, F]:
    for U in [P, F]:
        for A in [P, F]:
            for S in [P, F]:
                for B in [P, F]:
                    p[W, U, A, S,B] = pW[W]*pU[U]*pA[W,U,A]*pSA[A,S]*pBA[A,B]

PP1 = sum([p[W, U, P, P , P] for W, U in product([P,F], [P,F])])
Rozne = sum([p[W, U, A, P , P] for W,U,A in product([P,F], [P,F], [P,F])])
Zad1exact = PP1/Rozne
print('EXACT - Prawdopodobieństwo wystąpienia alarmu : ', Zad1exact)

PP2 = sum([p[P, U, A, P , P] for U, A in product([P,F], [P,F])])
Rozne = sum([p[W, U, A, P , P] for W,U,A in product([P,F], [P,F], [P,F])])
Zad2exact = PP2/Rozne
print('EXACT - Prawdopodobieństwo włamania: ', Zad2exact)


I = 100 
K = 100000 

avg_pw_b_k = 0
avg_pw_k_b = 0

plt.axis([0, I, 0, 1])
plt.xlabel('Iteracja')
plt.ylabel('Prawdopodobieństwo')
plt.plot([0, I], [Zad1exact, Zad1exact], color='b')
plt.plot([0, I], [Zad2exact, Zad2exact], color='b')

np.random.seed(1)
for i in range(1, I):

    W = np.random.random(K) < pW[P]
    U = np.random.random(K) < pU[P]
    APP = np.random.random(K) < pA[P, P, P]
    APF = np.random.random(K) < pA[P, F, P]
    AFP = np.random.random(K) < pA[F, P, P]
    AFF = np.random.random(K) < pA[F, F, P]
    SP = np.random.random(K) < pSA[P, P]
    SF = np.random.random(K) < pSA[F, P]
    BP = np.random.random(K) < pBA[P, P]
    BF = np.random.random(K) < pBA[F, P]
    A = np.logical_or.reduce((
        np.logical_and.reduce((W, U, APP)),
        np.logical_and.reduce((W, np.logical_not(U), APF)),
        np.logical_and.reduce((np.logical_not(W), U, AFP)),
        np.logical_and.reduce((np.logical_not(W), np.logical_not(U), AFF)),
    ))
    S = np.logical_or(
        np.logical_and(A, SP),
        np.logical_and(np.logical_not(A), SF)
    )
    B = np.logical_or(
        np.logical_and(A, BP),
        np.logical_and(np.logical_not(A), BF)
    )

    pw_b_k__mc = np.sum(np.logical_and(np.logical_and(A,B),S)) / np.sum(np.logical_and(B,S))
    pw_k_b__mc = np.sum(np.logical_and(np.logical_and(S,W),B)) / np.sum(np.logical_and(B,S))


    avg_pw_b_k = avg_pw_b_k + (pw_b_k__mc - avg_pw_b_k) / i
    avg_pw_k_b = avg_pw_k_b + (pw_k_b__mc - avg_pw_k_b) / i

    plt.scatter(i, avg_pw_b_k, marker='.', s=1, color='r')
    plt.scatter(i, avg_pw_k_b, marker='.', s=1, color='r')

    plt.pause(0.001)

print('Monte Carlo - Prawdopodobieństwo wystąpienia alarmu : ', avg_pw_b_k)
print('Monte Carlo - Prawdopodobieństwo włamania: ', avg_pw_k_b)



