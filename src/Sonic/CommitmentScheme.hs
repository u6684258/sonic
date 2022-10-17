-- Polynomial commitment scheme inspired by Kate et al.
{-# LANGUAGE RecordWildCards #-}
module Sonic.CommitmentScheme
  ( commitPoly
  , openPoly
  , pcV
  , PcVOutput
  , pcVShow
  ) where

import Protolude hiding (quot)
import qualified Data.Vector as V
import Data.Curve (Curve(..), mul)
import Data.Euclidean (divide)
import Data.Maybe (fromJust)
import Data.Pairing.BN254 (Fr, G1, G2, GT, BN254, pairing)
import Data.Poly.Sparse.Laurent (VLaurent, eval, monomial)
import qualified GHC.Exts
import Sonic.SRS (SRS(..))

data PcVOutput = PcVOutput
  { difference :: Int
  , commitment :: G1 BN254
  , idx :: Fr
  , value :: Fr
  , proof :: G1 BN254
  , hxi :: G2 BN254
  , hax0 :: G2 BN254
  , hax1 :: G2 BN254
  } deriving (Eq, Show)

-- Commit(info, f(X)) -> F:
commitPoly
  :: SRS           -- srs
  -> Int           -- max
  -> VLaurent Fr   -- f(X)
  -> G1 BN254   -- F
commitPoly SRS{..} maxm fX
  = foldl' (\acc (e, v)  -> acc <> if e > 0
             then (index "commitPoly: gPositiveAlphaX" gPositiveAlphaX (e - 1)) `mul` v      -- {g^{alpha*X^{d-max}*f(X) : X<0}
             else (index "commitPoly: gNegativeAlphaX" gNegativeAlphaX (abs e - 1)) `mul` v  -- {g^{alpha*X^{d-max}*f(X) : X>0}
           ) mempty xfX
  where
    difference = srsD - maxm            -- d-max
    powofx = monomial difference 1      -- X^{d-max}
    xfX = GHC.Exts.toList (powofx * fX) -- X^{d-max} * f(X)

-- Open(info, F, z, f(X)) -> (f(z), W):
openPoly
  :: SRS                 -- srs
  -> Fr                  -- z
  -> VLaurent Fr         -- f(X)
  -> (Fr, G1 BN254)   -- (f(z), W)
openPoly SRS{..} z fX = (fz, w)
  where
    fz = eval fX z -- f(z)
    wPoly = fromJust $ (fX - monomial 0 fz) `divide` GHC.Exts.fromList [(0, -z), (1, 1)] -- w(X) = (f(X) - f(z))/(X-z)
    w = foldl' (\acc (e, v) -> acc <> if e >= 0
                 then (index "openPoly: gPositiveX" gPositiveX e) `mul` v           -- {g^{w(X)} : X>=0}
                 else (index "openPoly: gNegativeX" gNegativeX (abs e - 1)) `mul` v -- {g^{w(X)} : X<0}
               ) mempty (GHC.Exts.toList wPoly)

-- pcV(info, F, z, (v, W)) -> 0|1:
pcV
  :: SRS                -- srs
  -> Int                -- max
  -> G1 BN254        -- F
  -> Fr                 -- z
  -> (Fr, G1 BN254)  -- (f(z), W)
  -> Bool               -- 0|1
pcV SRS{..} maxm commitment z (v, w)
  = eA <> eB == eC
  where
    eA, eB, eC :: GT BN254
    eA = pairing w (hPositiveAlphaX V.! 1)                                     -- e(W, h^{alpha*x})
    eB = pairing ((gen `mul` v) <> (w `mul` negate z)) (hPositiveAlphaX V.! 0) -- e(g^v W^{-z}, h^{alpha})
    eC = pairing commitment hxi                                                -- e(F, h^{x^{-d+max}})
    difference = -srsD + maxm                        -- -d+max
    hxi = if difference >= 0                         -- h^{x^{-d+max}}
          then index "pcV: hPositiveX" hPositiveX difference
          else index "pcV: hNegativeX" hNegativeX (abs difference - 1)

-- pcVShow(info, F, z, (v, W)) -> 0|1:
pcVShow
  :: SRS                -- srs
  -> Int                -- max
  -> G1 BN254        -- F
  -> Fr                 -- z
  -> (Fr, G1 BN254)  -- (f(z), W)
  -> PcVOutput            
pcVShow SRS{..} maxm commitment z (v, w)
  = PcVOutput{
    difference = difference
  , commitment = commitment
  , idx = z
  , value = v
  , proof = w
  , hxi = hxi
  , hax0 = eA
  , hax1 = eB
  }
  where
    eA, eB, hxi :: G2 BN254
    eA = hPositiveAlphaX V.! 1                 -- h^{alpha*x}
    eB = hPositiveAlphaX V.! 0                  -- h^{alpha}
    difference = -srsD + maxm                        -- -d+max
    hxi = if difference >= 0                         -- h^{x^{-d+max}}
          then index "pcV: hPositiveX" hPositiveX difference
          else index "pcV: hNegativeX" hNegativeX (abs difference - 1)

index :: Text -> V.Vector a -> Int -> a
index annot v e = fromMaybe err (v V.!? e)
  where
    err = panic $ annot <> " is not long enough: " <> show e <> " >= " <> show (V.length v)

