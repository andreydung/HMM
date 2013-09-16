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

    def evaluate(self, observations):
        """ Likelihood of a sequence
            Forward backward procedure
        """
        alpha=self.pi*self.B[:,observations[0]]

        for i in range(1,len(observations)):
            alpha=np.dot(alpha,self.A)*self.B[:,observations[i]]
        return sum(alpha)

    def decoding(self, observations):
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

    def display(self):
        # print self.A
        # print self.B
        print self.pi

    def train(self, observations, criterion):
        """ Baum Welch estimation
        """
        nStates = self.A.shape[0]
        nSamples = len(observations)
        converge=False
        counter=0
        while not converge and counter<10:
            counter+=1
            # ==================== Expectation step ======================
            #scale factors
            c = np.zeros(nSamples) 

            # Forward variable
            alpha=np.zeros((nStates, nSamples))
            alpha[:,0] = self.pi*self.B[:,observations[0]]
            c[0] = 1.0/np.sum(alpha[:,0])
            alpha[:,0] = c[0] * alpha[:,0]
            for t in range(1,nSamples):
                alpha[:,t] = np.dot(alpha[:, t-1],self.A)*self.B[:,observations[t]]
                c[t] = 1.0/np.sum(alpha[:,t])
                alpha[:,t] = c[t] * alpha[:,t]

            # Backward variable
            beta = np.zeros((nStates,nSamples))
            beta[:,nSamples-1] = 1
            beta[:,nSamples-1] = c[nSamples-1] * beta[:,nSamples-1]
            for t in range(len(observations)-1,0,-1):
                beta[:,t-1] = np.dot(self.A, (self.B[:,observations[t]] * beta[:,t]))
                beta[:,t-1] = c[t-1] * beta[:,t-1]
            
            # Xi and gamma variables
            xi = np.zeros((nSamples-1,nStates,nStates))
            for t in range(nSamples-1):
                denom = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,observations[t+1]].T, beta[:,t+1])
                for i in range(nStates):
                    for j in range(nStates):
                        numer = alpha[i,t] * self.A[i,j] * self.B[j,observations[t+1]] * beta[j,t+1]
                        xi[t, i,j] = numer / denom

            gamma=np.sum(xi,axis=2)
            prod =  (alpha[:,nSamples-1] * beta[:,nSamples-1]).reshape((-1,1)).T[0]
            gamma = np.vstack((gamma,  prod / np.sum(prod)))


            # ================ Maximization step ========================
            # Update HMM model parameters
            newpi = gamma[0,:]
            newA = np.sum(xi, axis=0)/ np.sum(gamma, axis=0).T
            
            newB = B
            nLevels = self.B.shape[1]
            sumgamma = np.sum(gamma,axis=0)
            
            for lev in range(nLevels):
                mask = observations == lev
                newB[:,lev] = np.sum(gamma[mask,:],axis=0) / sumgamma
            
            # Check if converge
            if np.max(abs(pi - newpi)) < criterion and \
                   np.max(abs(A - newA)) < criterion and \
                   np.max(abs(B - newB)) < criterion:
                converge=True
  
            self.pi=newpi
            self.A=newA
            self.B=newB

            HMM.display(self)

if __name__ == '__main__':
    #'Two states, three possible observations in a state'
    pi = np.array([0.5, 0.5])
    A = np.array([[0.7, 0.3],
                    [0.4, 0.6]])
    B = np.array([[0.1, 0.4, 0.5],
                  [0.7, 0.2, 0.1]])

    hmm=HMM(A,B,pi)
    obs=np.array([0,1,0,2,1,1,1])
    hmm.train(obs,0.1)
    

