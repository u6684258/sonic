{-# LANGUAGE NamedFieldPuns #-}
module Main where

import Protolude
import Control.Monad.Random (getRandomR)
import Bulletproofs.ArithmeticCircuit
import Data.Pairing.BN254 (Fr)
import Data.Field.Galois (rnd)
import Data.List.Split (divvy)

import Sonic.SRS as SRS
import Sonic.Protocol
import Sonic.Circuits

import qualified Data.Text    as Text
import qualified Data.Text.IO as Text
import Data.Text.Read (decimal, signed)

import Data.Time

-- import qualified Data.ByteString
-- import qualified Crypto.Hash.SHA256 as SHA256

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
  start <- getCurrentTime
  (proof, rndOracle@RndOracle{..}, verifierData) <- prove srsRaw srsLocal assignment circuit
  stop <- getCurrentTime
  -- putText $ "proof: " <> show proof
  -- putText $ "rnds: " <> show rndOracle

  putText $ "success:" <> show (verify srsRaw srsLocal circuit proof rndOracleY rndOracleZ rndOracleYZs)
  print $ diffUTCTime stop start

  writeFile "output/proof.txt" $ show $ proof
  writeFile "output/rndOracle.txt" $ show $ rndOracle
  writeFile "output/srsRaw.txt" $ show $ srsRaw
  writeFile "output/srsLocal.txt" $ show $ srsLocal
  writeFile "output/verifierData.txt" $ show $ verifierData
  where
    -- n: Number of multiplication constraints
    n = length $ aL assignment
    randomD n = getRandomR (7 * n, 100 * n)


runExample :: IO ()
runExample = do
  pX <- rnd

  wLS <- fmap Text.words (Text.readFile "input/sample_wL.txt")
  wRS <- fmap Text.words (Text.readFile "input/sample_wR.txt")
  wOS <- fmap Text.words (Text.readFile "input/sample_wO.txt")
  csS <- fmap Text.words (Text.readFile "input/sample_cs.txt")
  aLS <- fmap Text.words (Text.readFile "input/sample_aL.txt")
  aRS <- fmap Text.words (Text.readFile "input/sample_aR.txt")
  aOS <- fmap Text.words (Text.readFile "input/sample_aO.txt")
  let wLL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wLS))
      wRL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wRS))
      wOL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wOS))
      cs = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) csS))
      aL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aLS))
      aR = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aRS))
      aO = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aOS))
      wL = divvy 2 2 wLL
      wR = divvy 2 2 wRL
      wO = divvy 2 2 wOL

      (arithCircuit, assignment) = arithCircuitExample wL wR wO cs aL aR aO
  -- success <- sonicProtocol arithCircuit assignment pX
  -- putText $ "Success: " <> show success
  outputProof arithCircuit assignment pX



main :: IO ()
main = runExample
-- main = do
--   wLS <- fmap Text.words (Text.readFile "input/sample_wL.txt")
--   wRS <- fmap Text.words (Text.readFile "input/sample_wR.txt")
--   wOS <- fmap Text.words (Text.readFile "input/sample_wO.txt")
--   csS <- fmap Text.words (Text.readFile "input/sample_cs.txt")
--   aLS <- fmap Text.words (Text.readFile "input/sample_aL.txt")
--   aRS <- fmap Text.words (Text.readFile "input/sample_aR.txt")
--   aOS <- fmap Text.words (Text.readFile "input/sample_aO.txt")
--   let wLL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wLS))
--       wRL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wRS))
--       wOL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wOS))
--       cs = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) csS))
--       aL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aLS))
--       aR = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aRS))
--       aO = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aOS))
--       wL = divvy 2 2 wLL
--       wR = divvy 2 2 wRL
--       wO = divvy 2 2 wOL
--   print wL
--   print wR
--   print wO
--   print cs
--   print aL
--   print aR
--   print aO
  

