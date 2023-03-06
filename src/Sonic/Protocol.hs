-- The interactive Sonic protocol to check that the prover knows a valid assignment of the wires in the circuit

{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
module Sonic.Protocol
  ( Proof(..)
  , RndOracle(..)
  , Polys(..)
  -- , VerifierSRSHString
  -- , getverifyHxiString
  , VerifierData(..)
  , prove
  , verify
  , proveOutsourced
  ) where

import Protolude hiding (head)
import Data.List (head)
-- import qualified Data.Vector as V
import Data.Pairing.BN254 (Fr, G1, G2, BN254)
import Control.Monad.Random (MonadRandom)
import Bulletproofs.ArithmeticCircuit (ArithCircuit(..), GateWeights(..), Assignment(..)) --, GateWeights(..)
import Data.Field.Galois (rnd)
import Data.Poly.Sparse.Laurent (VLaurent, monomial) -- , eval
-- import qualified GHC.Exts

import Sonic.SRS (SRS(..))
import Sonic.Constraints (rPoly, rPolyRaw, tPoly, sPoly, kPoly) --
import Sonic.CommitmentScheme (commitPoly, openPoly, pcV) --, openPoly, pcV, pcVGetHxi
-- import Sonic.Signature (HscProof(..), hscProve, hscVerify) -- 
import Sonic.Utils (evalY, evalX, BiVLaurent)


data Proof = Proof
  { commitRLocal :: G1 BN254
  , commitRRaw :: G1 BN254
  , commitRAll :: G1 BN254
  , prT :: G1 BN254
  , prA :: Fr
  , prWa :: G1 BN254
  , prB :: Fr
  , prWb :: G1 BN254
  , prARaw :: Fr
  , prWaRaw :: G1 BN254
  , prALocal :: Fr
  , prWaLocal :: G1 BN254
  , prWt :: G1 BN254
  -- , proofOutsourced :: ProofOutsourced
  -- , prHscProof :: HscProof
  } deriving (Eq, Show, Generic, NFData)

data Polys = Polys
  {
    polyS :: BiVLaurent Fr
  , polyR :: BiVLaurent Fr
  , polyR1Raw :: VLaurent Fr
  , polyR1Local :: VLaurent Fr
  , polyR1 :: VLaurent Fr
  , polyT :: BiVLaurent Fr
  , polyK :: VLaurent Fr
  }  deriving (Eq, Show, Generic, NFData)

data ProofOutsourced = ProofOutsourced
  { prS :: Fr
  , commitSX :: G1 BN254
  , commitSY :: G1 BN254
  , commitK :: G1 BN254
  , sXz :: Fr
  , wsXz :: G1 BN254
  , sX1 :: Fr
  , wsX1 :: G1 BN254
  , sYy :: Fr
  , wsYy :: G1 BN254
  , ky :: Fr
  , wky :: G1 BN254
  -- , prHscProof :: HscProof
  } deriving (Eq, Show, Generic, NFData)

-- | Values created non-interactively in the random oracle model during proof generation
data RndOracle = RndOracle
  { rndOracleY :: Fr
  , rndOracleZ :: Fr
  , rndOracleYZ :: Fr
  } deriving (Eq, Show, Generic, NFData)

data VerifierData = VerifierData
  { kY :: VLaurent Fr
  , rY :: BiVLaurent Fr
  , sXY :: BiVLaurent Fr
  , n :: Int
  } deriving (Eq, Show, Generic, NFData)

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
-- proofOutsourced
  
prove
  :: MonadRandom m
  => Int
  -> Int
  -- -> ProofOutsourced
  -- -> RndOracle
  -> Assignment Fr
  -> ArithCircuit Fr
  -> m (Polys)
prove upSize n assignment@Assignment{..} arithCircuit@ArithCircuit{..} =
  if 999999 < 3*n
    -- then panic $ "Parameter d is not large enough: " <> show (srsD srsLocal) <> " should be greater than " <>  show (7*n)
    then panic $ "NaN"
    else do

    -- cns <- replicateM 4 rnd                 -- c_{n+1}, c_{n+2}, c_{n+3}, c_{n+4} <- F_p
    -- let sumcXY :: BiVLaurent Fr             -- \sum_{i=1}^4 c_{n+i}X^{-2n-i}Y^{-2n-i}
    --     sumcXY = GHC.Exts.fromList $
    --       zipWith (\i cni -> (negate (2 * n + i), monomial (negate (2 * n + i)) cni)) [1..] cns
    let polyR = rPoly assignment
        polyRRaw = evalY 1 (rPolyRaw assignment upSize)  -- r(X, Y) <- r(X, Y) + \sum_{i=1}^4 c_{n+i}X^{-2n-i}Y^{-2n-i}
        polyRAll = evalY 1 polyR
        polyRLocal = polyRAll - polyRRaw
        polyRRawShift = polyRRaw * monomial (-upSize) 1
        -- commitR = commitPoly srsLocal (fromIntegral n) polyRLocal -- R <- Commit(bp,srs,n,r(X,1))
        -- commitRRaw = commitPoly srsRaw (fromIntegral n) polyRRaw
        -- commitRAll = commitPoly srsLocal (fromIntegral n) polyRAll
    -- zkV_1(info, R) -> y
    -- y <- rnd
    -- let y = rndOracleY

    -- zkP_2(y) -> T
    let kY = kPoly cs n                     -- k(Y)
        sXY = sPoly weights                 -- s(X, Y)
        tXY = tPoly polyR sXY kY           -- t(X, Y)
        -- tXy = evalY y tXY                   -- t(X, y)
        -- commitT = commitPoly srsLocal (srsD srsLocal) tXy   -- T

    -- zkV_2(T) -> z
    -- z <- rnd
    -- let z = rndOracleZ
    -- zkP_3(z) -> (a, W_a, b, W_b_, W_t, s, sc)
    -- let (aLocal, waLocal) = openPoly srsLocal z polyRLocal       -- (a=r(z,1),W_a) <- Open(R,z,r(X,1))
    --     (aRaw, waRaw) = openPoly srsRaw z polyRRaw        -- (a=r(z,1),W_a) <- Open(R,z,r(X,1))
    --     (a, wa) = openPoly srsLocal z polyRAll       -- (a=r(z,1),W_a) <- Open(R,z,r(X,1))
    --     (b, wb) = openPoly srsLocal (y * z) polyRAll  -- (b=r(z,y),W_b) <- Open(R,yz,r(X,1))
    --     (_, wt) = openPoly srsLocal z (evalY y tXY)            -- (_=t(z,y),W_a) <- Open(T,z,t(X,y))
        -- szY = evalX z sXY
        -- commitC = commitPoly srsLocal (srsD srsLocal) szY   -- C
        -- (c, wc) = openPoly srsLocal y szY      -- c = Open(C, y, s′(z, Y ))      
        -- (cOld, wcOld) = openPoly srsLocal yOld szY      -- cold = Open(C, yold, s′(z, Y ))
        -- (s, ws) = openPoly srsLocal z sXy      -- s = Open(S, z, s′(X, y))
        -- (sOld, wsOld) = openPoly srsLocal z sXOldy      -- sold = Open(Sold, z, s′(X, y))

    -- zkV_2(T) -> z
    -- yNew <- rnd
    -- let sXyNew = evalY yNew sXY
    --     commitSNew = commitPoly srsLocal (srsD srsLocal) sXyNew    -- Snew = Cm(bp, srs, d, s′(X, ynew))
    --     (sNew, wsNew) = openPoly srsLocal z sXyNew  -- snew = Open(Snew, z, s′(X, ynew))
    --     (cNew, wcNew) = openPoly srsLocal yNew szY  -- cnew = Open(C, ynew, s′(z, Y ))

    -- let szy = eval (evalY y sXY) z                        -- s=s(z,y)
    -- hscProof <- hscProve srsLocal sXY yzs
    
    pure ( Polys
          --  { commitRLocal = commitR
          --  , commitRRaw = commitRRaw
          --  , commitRAll = commitRAll
          --  , prT = commitT
           { polyS = sXY
             , polyR = polyR
          --  , 
           , polyR1Raw = polyRRawShift
           , polyR1Local = polyRLocal
           , polyR1 = polyRAll
           , polyT = tXY
           , polyK = kY
          --  , prA = a
          --  , prWa = wa
          --  , prB = b
          --  , prWb = wb
          --  , prARaw = aRaw
          --  , prWaRaw = waRaw
          --  , prALocal = aLocal
          --  , prWaLocal = waLocal
          --  , prWt = wt
          --  , proofOutsourced = proofOutsourced
           }
        --  , rands
        --  , VerifierData
        --   { kY = kY
        --   , rY = polyR
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
  -> ProofOutsourced
  -> Fr
  -> Fr
  -> Fr
  -> Bool
verify srsRaw srsLocal ArithCircuit{..} Proof{..} ProofOutsourced{..} y z yz
  = let t = prA * (prB + prS) - ky
        checks = [ pcV srsRaw (fromIntegral n) commitRRaw z (prARaw, prWaRaw)
                 , pcV srsLocal (fromIntegral n) commitRLocal z (prALocal, prWaLocal)
                 , pcV srsLocal (fromIntegral n) commitRAll z (prA, prWa)
                 , pcV srsLocal (fromIntegral n) commitRAll (y * z) (prB, prWb)
                 , pcV srsLocal (srsD srsLocal) prT z (t, prWt)
                 , pcV srsLocal (srsD srsLocal) commitK y (ky, wky)
                 , pcV srsLocal (srsD srsLocal) commitSX z (sXz, wsXz)
                 , pcV srsLocal (srsD srsLocal) commitSX 1 (sX1, wsX1)
                 , pcV srsLocal (srsD srsLocal) commitSY y (sYy, wsYy)
                 , prA == prARaw + prALocal
                 , sX1 == sYy
                ]
    in and checks
  where
    n = length . head . wL $ weights
    -- kY = kPoly cs n
    -- sXY = sPoly weights

proveOutsourced
  :: MonadRandom m
  => SRS
  -> SRS
  -> Int
  -> Assignment Fr
  -> ArithCircuit Fr
  -> m (ProofOutsourced, RndOracle)
proveOutsourced srsRaw srsLocal upSize assignment@Assignment{..} arithCircuit@ArithCircuit{..} =
  if srsD srsLocal < 3*n
    then panic $ "Parameter d is not large enough: " <> show (srsD srsLocal) <> " should be greater than " <>  show (7*n)
    else do

    y <- rnd

    -- zkP_2(y) -> T
    let kY = kPoly cs n                     -- k(Y)
        sXY = sPoly weights                 -- s(X, Y)
        sY = evalX 1 sXY
        commitSY = commitPoly srsLocal (srsD srsLocal) sY   -- SY        -- sXy = evalY y sXY
        sX = evalY y sXY
        commitSX = commitPoly srsLocal (srsD srsLocal) sX   -- S
        commitK = commitPoly srsLocal (srsD srsLocal) kY   -- T

    -- zkV_2(T) -> z
    z <- rnd

    let (k, wk) = openPoly srsLocal y kY
        (sXz, wsXz) = openPoly srsLocal z sX -- snew = Open(Snew, z, s′(X, ynew))
        (sX1, wsX1) = openPoly srsLocal 1 sX -- snew = Open(Snew, z, s′(X, ynew))
        (sYy, wsYy) = openPoly srsLocal y sY -- snew = Open(Snew, z, s′(X, ynew))

    -- let szy = eval sX z                        -- s=s(z,y)
    -- hscProof <- hscProve srsLocal sXY yzs
    
    pure ( ProofOutsourced
           {  prS = sXz
            , commitSX = commitSX
            , commitSY = commitSY
            , commitK = commitK
            , sXz = sXz
            , wsXz = wsXz
            , sX1 = sX1
            , wsX1 = wsX1
            , sYy = sYy
            , wsYy = wsYy
            , ky = k
            , wky = wk
           }
         , RndOracle
           { rndOracleY = y
           , rndOracleZ = z
           , rndOracleYZ = y * z
           }
         )
  where
    n :: Int
    n = length aL
    -- m :: Int
    -- m = length . wL $ weights
    -- m = 1



-- getverifyHxiString
--   :: SRS
--   -> SRS
--   -> ArithCircuit Fr
--   -> Proof
--   -> Fr
--   -> Fr
--   -> [(Fr, Fr)]
--   -> VerifierSRSHString
-- getverifyHxiString srsRaw srsLocal ArithCircuit{..} Proof{..} y z yzs
--   = VerifierSRSHString
--   {
--     rawAhaxeA = (hPositiveAlphaX srsRaw) V.! 1
--   , rawAhaxeB = (hPositiveAlphaX srsRaw) V.! 0
--   , locAhaxeA = (hPositiveAlphaX srsLocal) V.! 1
--   , locAhaxeB = (hPositiveAlphaX srsLocal) V.! 0
--   , rawAhxeC  = pcVGetHxi srsRaw (fromIntegral n)
--   , locAhxeC  = pcVGetHxi srsLocal (fromIntegral n)
--   , thxeC     = pcVGetHxi srsLocal (srsD srsLocal)
--   , keval  = ky
--   , teval = (prA+prARaw) * ((prB+prBRaw) + s) - k
--   }
--   where
--     ky  = eval kY y
--     n = length . head . wL $ weights
--     kY = kPoly cs n             