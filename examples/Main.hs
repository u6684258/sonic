{-# LANGUAGE NamedFieldPuns #-}
module Main where

import Protolude
-- import Control.Monad.Random (getRandomR)
import Bulletproofs.ArithmeticCircuit
import Data.Pairing.BN254 (Fr)
-- import Data.Field.Galois (rnd)
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


outputProof:: ArithCircuit Fr -> Assignment Fr -> Fr -> Fr -> Fr -> Fr -> IO ()
outputProof circuit assignment pXRaw pXLocal alphaRaw alphaLocal = do
  -- Setup for an SRS

  srsRaw <- SRS.new <$> pure n <*> pure pXRaw <*> pure alphaRaw
  srsLocal <- SRS.new <$> pure n <*> pure pXLocal <*> pure alphaLocal

  print $ alphaRaw
  print $ alphaLocal
  print $ (srsD srsRaw)
  print $ (srsD srsLocal)
  print $ pXRaw
  print $ pXLocal
  -- Prover
  start <- getCurrentTime
  (proof, rndOracle@RndOracle{..}, verifierData) <- prove srsRaw srsLocal 2 assignment circuit
  stop <- getCurrentTime
  -- putText $ "proof: " <> show proof
  -- putText $ "rnds: " <> show rndOracle
  print $ diffUTCTime stop start

  startVer <- getCurrentTime
  putText $ "success:" <> show (verify srsRaw srsLocal circuit proof rndOracleY rndOracleZ rndOracleYZs)
  stopVer <- getCurrentTime
  print $ diffUTCTime stopVer startVer
  writeFile "output/proof.txt" $ show $ proof
  writeFile "output/rndOracle.txt" $ show $ rndOracle
  writeFile "output/srsRaw.txt" $ show $ srsRaw
  writeFile "output/srsLocal.txt" $ show $ srsLocal
  writeFile "output/verifierData.txt" $ show $ verifierData
  where
    -- n: Number of multiplication constraints
    n = (length $ aL assignment) * 10
    


runExample :: IO ()
runExample = do

  let alphaRaw = 4537460542209314651160888417413866091249215769242027952878258319870902529429
      alphaLocal = 10318208573435976958553514324035131570702768863559211839926632048062238413316
      pXRaw = 4518563069472097478295524977775021906947577384653869551543466909390271555451
      pXLocal = 19708723214916757413126173169122466312825114221484651297201668130676937834219

  wLS <- fmap Text.words (Text.readFile "input/wL.txt")
  wRS <- fmap Text.words (Text.readFile "input/wR.txt")
  wOS <- fmap Text.words (Text.readFile "input/wO.txt")
  csS <- fmap Text.words (Text.readFile "input/cs.txt")
  aLS <- fmap Text.words (Text.readFile "input/aL.txt")
  aRS <- fmap Text.words (Text.readFile "input/aR.txt")
  aOS <- fmap Text.words (Text.readFile "input/aO.txt")
  let wLL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wLS))
      wRL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wRS))
      wOL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) wOS))
      cs = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) csS))
      aL = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aLS))
      aR = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aRS))
      aO = foldr (\x acc -> (fst x):acc) [] (rights (map (signed decimal) aOS))
      wL = divvy 50 50 wLL
      wR = divvy 50 50 wRL
      wO = divvy 50 50 wOL

      (arithCircuit, assignment) = arithCircuitExample wL wR wO cs aL aR aO
  -- success <- sonicProtocol arithCircuit assignment pX
  -- putText $ "Success: " <> show success
  outputProof arithCircuit assignment pXRaw pXLocal alphaRaw alphaLocal



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
  

