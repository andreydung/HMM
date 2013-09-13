import numpy as np
import math

class HMM:
    def __init__(self, A, B, pi):
        assert A.shape[1]==B.shape[0], "Dimension between A and B disagree"
        assert pi.shape[0]==A.shape[0], "Dimension between pi and A disagree"
        self.A=A
        self.B=B
        self.pi=pi

    def simulate(self,nSteps):

        def drawFrom(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]

        observations = np.zeros(nSteps)
        states = np.zeros(nSteps)
        states[0] = drawFrom(self.pi)
        observations[0] = drawFrom(self.B[states[0],:])
        for t in range(1,nSteps):
            states[t] = drawFrom(self.A[states[t-1],:])
            observations[t] = drawFrom(self.B[states[t],:])
        return observations,states

    def forwardbackward(self, observations):
        """ Likelihood of a sequence
            Forward backward procedure
        """
        alpha=self.pi*self.B[:,observations[0]]

        for i in range(1,len(observations)):
            alpha=np.dot(alpha,self.A)*self.B[:,observations[i]]
            print alpha
        return sum(alpha)

    def findstate(self, observations):
        """ Find hidden state of HMM from observations
        Viterbi algorithm
        """
        delta = self.pi*self.B[:,observations[0]]
        states=np.zeros(len(observations))
        states[0] = 0
        # print np.log(delta)/np.log(2)

        for i in range(1,len(observations)):
            delta = np.dot(np.diagflat(delta), A).max(axis=0) * self.B[:,observations[i]]
            states[i]= delta.argmax(axis=0)
            # print np.log(delta)/np.log(2)
        return states

    def hmmestimate(self):
        """ Baum Welch estimation
        """

 

if __name__ == '__main__':
    #'Two states, three possible observations in a state'
    pi = np.array([0.25, 0.5, 0.25])
    A = np.array([[0.25, 0.5, 0.25],
                      [0.25, 0.25, 0.5],
                      [0.5, 0.5, 0]])
    B = np.array([[1,0,0,0],
                  [0.25,0.5,0,0.25],
                  [0.25, 0.25, 0.25, 0.25]])

    hmm=HMM(A,B,pi)
    print hmm.findstate(np.array([1,1,2]))