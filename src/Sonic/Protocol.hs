-- The interactive Sonic protocol to check that the prover knows a valid assignment of the wires in the circuit

{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
module Sonic.Protocol
  ( Proof(..)
  , RndOracle(..)
  , VerifierSRSHString
  , getverifyHxiString
  , prove
  , verify
  ) where

import Protolude hiding (head)
import Data.List (head)
import qualified Data.Vector as V
import Data.Pairing.BN254 (Fr, G1, G2, BN254)
import Control.Monad.Random (MonadRandom)
import Bulletproofs.ArithmeticCircuit (ArithCircuit(..), Assignment(..), GateWeights(..))
import Data.Field.Galois (rnd)
import Data.Poly.Sparse.Laurent (monomial, eval)
import qualified GHC.Exts

import Sonic.SRS (SRS(..))
import Sonic.Constraints (rPoly, rPolyRaw, sPoly, tPoly, kPoly)
import Sonic.CommitmentScheme (commitPoly, openPoly, pcV, pcVGetHxi)
-- import Sonic.Signature (HscProof(..), hscProve, hscVerify) -- 
import Sonic.Utils (evalY, evalX, BiVLaurent)

data Proof = Proof
  { prR :: G1 BN254
  , prRRaw :: G1 BN254
  , prT :: G1 BN254
  , prA :: Fr
  , prWa :: G1 BN254
  , prARaw :: Fr
  , prWaRaw :: G1 BN254
  , prB :: Fr
  , prWb :: G1 BN254
  , prBRaw :: Fr
  , prWbRaw :: G1 BN254
  , prWt :: G1 BN254
  -- , prS :: Fr
  , commitS :: G1 BN254
  , commitSOld :: G1 BN254
  , commitSNew :: G1 BN254
  , commitK :: G1 BN254
  , k :: Fr
  , commitC :: G1 BN254
  , c :: Fr
  , cOld :: Fr
  , cNew :: Fr
  , s :: Fr
  , sOld :: Fr
  , sNew :: Fr
  , wc :: G1 BN254
  , wcOld :: G1 BN254
  , wcNew :: G1 BN254
  , ws :: G1 BN254
  , wsOld :: G1 BN254
  , wsNew :: G1 BN254
  , wk :: G1 BN254
  -- , prHscProof :: HscProof
  } deriving (Eq, Show, Generic, NFData)

-- | Values created non-interactively in the random oracle model during proof generation
data RndOracle = RndOracle
  { rndOracleY :: Fr
  , rndOracleZ :: Fr
  , rndOracleYZ :: Fr
  , rndOracleyOld :: Fr
  , rndOracleyNew :: Fr
  } deriving (Eq, Show, Generic, NFData)

-- data VerifierData = VerifierData
--   { kY :: VLaurent Fr
--   , sXY :: BiVLaurent Fr
--   , n :: Int
--   } deriving (Eq, Show, Generic, NFData)

data VerifierSRSHString = VerifierSRSHString
  {
    rawAhaxeA :: G2 BN254
  , rawAhaxeB :: G2 BN254
  , locAhaxeA :: G2 BN254
  , locAhaxeB :: G2 BN254
  , rawAhxeC  :: G2 BN254
  , locAhxeC  :: G2 BN254
  , thxeC  :: G2 BN254
  , keval  :: Fr
  , teval  :: Fr
  } deriving (Eq, Show)

prove
  :: MonadRandom m
  => SRS
  -> SRS
  -> Int
  -> Assignment Fr
  -> ArithCircuit Fr
  -> m (Proof, RndOracle)
prove srsRaw srsLocal upSize assignment@Assignment{..} arithCircuit@ArithCircuit{..} =
  if srsD srsLocal < 3*n
    then panic $ "Parameter d is not large enough: " <> show (srsD srsLocal) <> " should be greater than " <>  show (7*n)
    else do
    yOld <- rnd
    -- ys <- replicateM m rnd
    -- zs <- replicateM m rnd
    -- let yzs = zip ys zs
    --     y = head ys
    --     z = head zs
    -- zkP_1(info,a,b,c) -> R
    cns <- replicateM 4 rnd                 -- c_{n+1}, c_{n+2}, c_{n+3}, c_{n+4} <- F_p
    let sumcXY :: BiVLaurent Fr             -- \sum_{i=1}^4 c_{n+i}X^{-2n-i}Y^{-2n-i}
        sumcXY = GHC.Exts.fromList $
          zipWith (\i cni -> (negate (2 * n + i), monomial (negate (2 * n + i)) cni)) [1..] cns
        polyRRaw = rPolyRaw assignment upSize  -- r(X, Y) <- r(X, Y) + \sum_{i=1}^4 c_{n+i}X^{-2n-i}Y^{-2n-i}
        polyR' = rPoly assignment + sumcXY 
        polyRLocal = polyR' - polyRRaw
        commitR = commitPoly srsLocal (fromIntegral n) (evalY 1 polyRLocal) -- R <- Commit(bp,srs,n,r(X,1))
        commitRRaw = commitPoly srsRaw (fromIntegral n) (evalY 1 polyRRaw)
    -- zkV_1(info, R) -> y
    y <- rnd

    -- zkP_2(y) -> T
    let kY = kPoly cs n                     -- k(Y)
        sXY = sPoly weights                 -- s(X, Y)
        tXY = tPoly polyR' sXY kY           -- t(X, Y)
        tXy = evalY y tXY                   -- t(X, y)
        sXy = evalY y sXY
        sXOldy = evalY yOld sXY
        commitT = commitPoly srsLocal (srsD srsLocal) tXy   -- T
        commitS = commitPoly srsLocal (srsD srsLocal) sXy   -- S
        commitSOld = commitPoly srsLocal (srsD srsLocal) sXOldy
        commitK = commitPoly srsLocal (srsD srsLocal) kY   -- T
        (k, wk) = openPoly srsLocal y kY

    -- zkV_2(T) -> z
    z <- rnd

    -- zkP_3(z) -> (a, W_a, b, W_b_, W_t, s, sc)
    let (aLocal, waLocal) = openPoly srsLocal z (evalY 1 polyRLocal)        -- (a=r(z,1),W_a) <- Open(R,z,r(X,1))
        (aRaw, waRaw) = openPoly srsRaw z (evalY 1 polyRRaw)        -- (a=r(z,1),W_a) <- Open(R,z,r(X,1))
        (bLocal, wbLocal) = openPoly srsLocal (y * z) (evalY 1 polyRLocal)  -- (b=r(z,y),W_b) <- Open(R,yz,r(X,1))
        (bRaw, wbRaw) = openPoly srsRaw (y * z) (evalY 1 polyRRaw)  -- (b=r(z,y),W_b) <- Open(R,yz,r(X,1))
        (_, wt) = openPoly srsLocal z (evalY y tXY)            -- (_=t(z,y),W_a) <- Open(T,z,t(X,y))
        szY = evalX z sXY
        commitC = commitPoly srsLocal (srsD srsLocal) szY   -- C
        (c, wc) = openPoly srsLocal y szY      -- c = Open(C, y, s′(z, Y ))      
        (cOld, wcOld) = openPoly srsLocal yOld szY      -- cold = Open(C, yold, s′(z, Y ))
        (s, ws) = openPoly srsLocal z sXy      -- s = Open(S, z, s′(X, y))
        (sOld, wsOld) = openPoly srsLocal z sXOldy      -- sold = Open(Sold, z, s′(X, y))

    -- zkV_2(T) -> z
    yNew <- rnd
    let sXyNew = evalY yNew sXY
        commitSNew = commitPoly srsLocal (srsD srsLocal) sXyNew    -- Snew = Cm(bp, srs, d, s′(X, ynew))
        (sNew, wsNew) = openPoly srsLocal z sXyNew  -- snew = Open(Snew, z, s′(X, ynew))
        (cNew, wcNew) = openPoly srsLocal yNew szY  -- cnew = Open(C, ynew, s′(z, Y ))

    -- let szy = eval (evalY y sXY) z                        -- s=s(z,y)
    -- hscProof <- hscProve srsLocal sXY yzs
    
    pure ( Proof
           { prR = commitR
           , prRRaw = commitRRaw
           , prT = commitT
           , prA = aLocal
           , prWa = waLocal
           , prARaw = aRaw
           , prWaRaw = waRaw
           , prB = bLocal
           , prWb = wbLocal
           , prBRaw = bRaw
           , prWbRaw = wbRaw
           , prWt = wt
          --  , prS = szy
            , commitS = commitS
            , commitSOld = commitSOld
            , commitSNew = commitSNew
            , commitK = commitK
            , k = k
            , commitC = commitC
            , c = c
            , cOld = cOld
            , cNew = cNew
            , s = s
            , sOld = sOld
            , sNew = sNew
            , wc = wc
            , wcOld = wcOld
            , wcNew = wcNew
            , ws = ws
            , wsOld = wsOld
            , wsNew = wsNew
            , wk = wk
           }
         , RndOracle
           { rndOracleY = y
           , rndOracleZ = z
           , rndOracleYZ = y * z
           , rndOracleyOld = yOld
            , rndOracleyNew = yNew
           }
        --  , VerifierData
        --   { kY = kY
        --   , sXY = sXY
        --   , n = n
        --   }
         )
  where
    n :: Int
    n = length aL
    -- m :: Int
    -- m = length . wL $ weights
    -- m = 1

verify
  :: SRS
  -> SRS
  -> ArithCircuit Fr
  -> Proof
  -> Fr
  -> Fr
  -> Fr
  -> Fr
  -> Fr
  -> Bool
verify srsRaw srsLocal ArithCircuit{..} Proof{..} y z yz yOld yNew
  = let t = (prA+prARaw) * ((prB+prBRaw) + s) - k
        checks = [ pcV srsRaw (fromIntegral n) prRRaw z (prARaw, prWaRaw)
                 , pcV srsLocal (fromIntegral n) prR z (prA, prWa)
                 , pcV srsRaw (fromIntegral n) prRRaw (y * z) (prBRaw, prWbRaw)
                 , pcV srsLocal (fromIntegral n) prR (y * z) (prB, prWb)
                 , pcV srsLocal (srsD srsLocal) prT z (t, prWt)
                 , pcV srsLocal (srsD srsLocal) commitK y (k, wk)
                 , pcV srsLocal (srsD srsLocal) commitC y (c, wc)
                 , pcV srsLocal (srsD srsLocal) commitC yOld (cOld, wcOld)
                 , pcV srsLocal (srsD srsLocal) commitC yNew (cNew, wcNew)
                 , pcV srsLocal (srsD srsLocal) commitS z (s, ws)
                 , pcV srsLocal (srsD srsLocal) commitSOld z (sOld, wsOld)
                 , pcV srsLocal (srsD srsLocal) commitSNew z (sNew, wsNew)
                --  , hscVerify srsLocal sXY yzs prHscProof
                ]
    in and checks
  where
    n = length . head . wL $ weights
    -- kY = kPoly cs n
    -- sXY = sPoly weights

getverifyHxiString
  :: SRS
  -> SRS
  -> ArithCircuit Fr
  -> Proof
  -> Fr
  -> Fr
  -> [(Fr, Fr)]
  -> VerifierSRSHString
getverifyHxiString srsRaw srsLocal ArithCircuit{..} Proof{..} y z yzs
  = VerifierSRSHString
  {
    rawAhaxeA = (hPositiveAlphaX srsRaw) V.! 1
  , rawAhaxeB = (hPositiveAlphaX srsRaw) V.! 0
  , locAhaxeA = (hPositiveAlphaX srsLocal) V.! 1
  , locAhaxeB = (hPositiveAlphaX srsLocal) V.! 0
  , rawAhxeC  = pcVGetHxi srsRaw (fromIntegral n)
  , locAhxeC  = pcVGetHxi srsLocal (fromIntegral n)
  , thxeC     = pcVGetHxi srsLocal (srsD srsLocal)
  , keval  = ky
  , teval = (prA+prARaw) * ((prB+prBRaw) + s) - k
  }
  where
    ky  = eval kY y
    n = length . head . wL $ weights
    kY = kPoly cs n             