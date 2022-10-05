{-# LANGUAGE NamedFieldPuns #-}
module Main where

import Protolude
import Control.Monad.Random (getRandomR)
import Bulletproofs.ArithmeticCircuit
import Data.Pairing.BN254 (Fr)
import Data.Field.Galois (rnd)

import Sonic.SRS as SRS
import Sonic.Protocol
import Sonic.Circuits

-- sonicProtocol :: SRS 
--   -> SRS 
--   -> Proof 
--   -> RndOracle 
--   -> ArithCircuit Fr 
--   -> Assignment Fr 
--   -> Fr 
--   -> IO Bool
-- sonicProtocol srsRaw srsLocal proof rndOracle@RndOracle{..} circuit assignment x = do
--   -- Verifier
--   pure $ verify srsRaw srsLocal circuit proof rndOracleY rndOracleZ rndOracleYZs


outputProof:: ArithCircuit Fr -> Assignment Fr -> Fr -> IO ()
outputProof circuit assignment x = do
  -- Setup for an SRS
  srsRaw <- SRS.new <$> randomD n <*> pure x <*> rnd
  srsLocal <- SRS.new <$> randomD n <*> pure x <*> rnd
  -- Prover
  (proof, rndOracle@RndOracle{..}) <- prove srsRaw srsLocal assignment circuit

  -- putText $ "proof: " <> show proof
  -- putText $ "rnds: " <> show rndOracle

  putText $ "success:" <> show (verify srsRaw srsLocal circuit proof rndOracleY rndOracleZ rndOracleYZs)
  
  writeFile "proof.txt" $ show $ proof
  writeFile "rndOracle.txt" $ show $ rndOracle
  writeFile "srsRaw.txt" $ show $ srsRaw
  writeFile "srsLocal.txt" $ show $ srsLocal
  where
    -- n: Number of multiplication constraints
    n = length $ aL assignment
    randomD n = getRandomR (7 * n, 100 * n)


runExample :: IO ()
runExample = do
  pX <- rnd
  pY <- rnd
  pZ <- rnd
  let (arithCircuit, assignment@Assignment{..}) = arithCircuitExample [pX,pY] pZ
  -- success <- sonicProtocol arithCircuit assignment pX
  -- putText $ "Success: " <> show success
  outputProof arithCircuit assignment pX

main :: IO ()
main = runExample
