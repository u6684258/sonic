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

-- sonicProtocol :: ArithCircuit Fr -> Assignment Fr -> Fr -> IO Bool
-- sonicProtocol circuit assignment x = do
--   -- Setup for an SRS
--   srs <- SRS.new <$> randomD n <*> pure x <*> rnd
--   -- Prover
--   (proof, rndOracle@RndOracle{..}) <- prove srs assignment circuit
--   -- Verifier
--   pure $ verify srs circuit proof rndOracleY rndOracleZ rndOracleYZs
--   where
--     -- n: Number of multiplication constraints
--     n = length $ aL assignment
--     randomD n = getRandomR (7 * n, 100 * n)

testCode:: ArithCircuit Fr -> Assignment Fr -> Fr -> IO ()
testCode circuit assignment x = do
  -- Setup for an SRS
  srsRaw <- SRS.new <$> randomD n <*> pure x <*> rnd
  srsLocal <- SRS.new <$> randomD n <*> pure x <*> rnd
  -- Prover
  (proof, rndOracle) <- prove srsRaw srsLocal assignment circuit

  putText $ "proof: " <> show proof
  putText $ "rnds: " <> show rndOracle
  where
    -- n: Number of multiplication constraints
    n = length $ aL assignment
    randomD n = getRandomR (7 * n, 100 * n)


runExample :: IO ()
runExample = do
  pX <- rnd
  pZ <- rnd
  let (arithCircuit, assignment@Assignment{..}) = arithCircuitExample pX pZ
  -- success <- sonicProtocol arithCircuit assignment pX
  -- putText $ "Success: " <> show success
  testCode arithCircuit assignment pX
  -- putText $ "proof: " <> show proof

main :: IO ()
main = runExample
