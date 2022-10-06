module Sonic.Circuits
  ( arithCircuitExample
  ) where


import Protolude
import Bulletproofs.ArithmeticCircuit
import Data.Pairing.BN254 (Fr)

-- 5 linear constraints (q = 5):
-- aO[0] = aO[1]
-- aL[0] = V[0] - z
-- aL[1] = V[2] - z
-- aR[0] = V[1] - z
-- aR[1] = V[3] - z
--
-- 2 multiplication constraints (implicit) (n = 2):
-- aL[0] * aR[0] = aO[0]
-- aL[1] * aR[1] = aO[1]
--
-- 4 input values (m = 4)

-- wL = [[0, 0]
--       ,[1, 0]
--       ,[0, 1]
--       ,[0, 0]
--       ,[0, 0]
--       ]
-- wR = [[0, 0]
--       ,[0, 0]
--       ,[0, 0]
--       ,[1, 0]
--       ,[0, 1]
--       ]
-- wO = [[1, -1]
--       ,[0, 0]
--       ,[0, 0]
--       ,[0, 0]
--       ,[0, 0]
--       ]
-- cs = [  0
--       , 4-pZ
--       , 9-pZ
--       , 9-pZ
--       , 4-pZ
--       ]
-- aL = [  4 - pZ
--       , 9 - pZ
--       ]
-- aR = [  9 - pZ
--       , 4 - pZ
--       ]
      
-- aO = zipWith (*) aL aR

-- T = 1
-- N = 1

arithCircuitExample :: [[Fr]]
                  -> [[Fr]] 
                  -> [[Fr]] 
                  -> [Fr]
                  -> [Fr]
                  -> [Fr]
                  -> [Fr]
                  -> (ArithCircuit Fr, Assignment Fr)
arithCircuitExample wL wR wO cs aL aR aO =
  let gateWeights = GateWeights wL wR wO
      assignment = Assignment aL aR aO
      circuit = ArithCircuit gateWeights witness cs
  in (circuit, assignment)