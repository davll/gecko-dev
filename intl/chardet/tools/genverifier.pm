#!/usr/local/bin/perl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

package genverifier;
use strict;
use vars qw(@ISA @EXPORT @EXPORT_OK $VERSION);

use Exporter;
$VERSION = 1.00;
@ISA = qw(Exporter);

@EXPORT       = qw(
                   GenVerifier
                  );
@EXPORT_OK    = qw();

sub GenNPL {
  my($ret) = << "END_MPL";
/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
END_MPL

  return $ret;
}

##--------------------------------------------------------------
sub GetClass {
  my($char, $clstbl) = @_;
  my($l);
  for($l =0; $l <= @$clstbl; $l++) {
    if(($clstbl->[$l][0] <= $char) && ($char <= $clstbl->[$l][1]))
    {
          return $clstbl->[$l][2];
    }
  }
  print "WARNING- there are no class for $char\n";
};
##--------------------------------------------------------------
sub GenClassPkg {
  my($name, $bits) = @_;
  return GenPkg($name, $bits, "_cls");
}
##--------------------------------------------------------------
sub GenStatePkg {
  my($name, $bits) = @_;
  return GenPkg($name, $bits, "_st");
};
##--------------------------------------------------------------
sub GenPkg {
  my($name, $bits, $tbl) = @_;
  my($ret);
  $ret = "    {\n" . 
         "       eIdxSft"  . $bits . "bits, \n" .
         "       eSftMsk"  . $bits . "bits, \n" . 
         "       eBitSft"  . $bits . "bits, \n" . 
         "       eUnitMsk" . $bits . "bits, \n" .
         "       " . $name . $tbl . " \n" . 
         "    }";
  return $ret;
};
##--------------------------------------------------------------
sub Gen4BitsClass {
  my($name, $clstbl) = @_;
  my($i,$j);
  my($cls);
  my($ret);
  $ret = "";
  $ret .= "static const uint32_t " . $name . "_cls [ 256 / 8 ] = {\n";
  for($i = 0; $i < 0x100; $i+= 8) {
     $ret .= "PCK4BITS(";
     for($j = $i; $j < $i + 8; $j++) {
         $cls = &GetClass($j,$clstbl);
         $ret .= sprintf("%d", $cls) ;
         if($j != ($i+7)) {
            $ret .= ",";
         }
     }
     if( $i+8 >= 0x100) {
        $ret .= ") ";
     } else {
        $ret .= "),";
     }
     $ret .= sprintf("  // %02x - %02x \n", $i, ($i+7));
  }
  $ret .= "};\n";
  return $ret;
};
##--------------------------------------------------------------
sub GenVerifier {
  my($name, $charset, $cls, $numcls, $st) = @_;
  my($ret);
  $ret = GenNPL();
  $ret .= GenNote();
  $ret .= GenHeader();
  $ret .= Gen4BitsClass($name, $cls);
  $ret .= "\n\n";
  $ret .= Gen4BitsState($name, $st);
  $ret .= "\n\n";
  $ret .= "static nsVerifier ns" . $name . "Verifier = {\n";
  $ret .= '     "' . $charset . '",' . "\n";
  $ret .= GenClassPkg($name, 4);
  $ret .= ",\n";
  $ret .= "    " . $numcls;
  $ret .= ",\n";
  $ret .= GenStatePkg($name, 4);
  $ret .= "\n};\n";
  return $ret;
 
};
##--------------------------------------------------------------
sub Gen4BitsState {
  my($name, $sttbl) = @_;
  my($lenafterpad) = (((@$sttbl-1) >> 3) + 1) << 3;
  my($i,$j);
  my($ret);
  $ret = "";
  $ret .= "static const uint32_t " . $name . "_st [ " . ($lenafterpad >> 3) . "] = {\n";
  for($i = 0; $i < $lenafterpad ; $i+= 8) {
     $ret .= "PCK4BITS(";
     for($j = $i; $j < $i + 8; $j++) {
         if(0 == $sttbl->[$j]) {
              $ret .= "eStart";
         } else { if(1 == $sttbl->[$j]) {
              $ret .= "eError";
         } else { if(2 == $sttbl->[$j]) {
              $ret .= "eItsMe";
         } else {
              $ret .= sprintf("     %d", $sttbl->[$j]) ;
         }}}
         if($j != ($i+7)) {
            $ret .= ",";
         }
     }
     if( $i+8 >= $lenafterpad ) {
        $ret .= ") ";
     } else {
        $ret .= "),";
     }
     $ret .= sprintf("//%02x-%02x \n", $i, ($i+7));
  }
  $ret .= "};\n";
  return $ret;
};
##--------------------------------------------------------------

sub GenNote {
  my($ret) = << "END_NOTE";
/* 
 * DO NOT EDIT THIS DOCUMENT MANUALLY !!!
 * THIS FILE IS AUTOMATICALLY GENERATED BY THE TOOLS UNDER
 *    mozilla/intl/chardet/tools/
 * Please contact ftang\@netscape.com or mozilla-i18n\@mozilla.org
 * if you have any question. Thanks
 */
END_NOTE
  return $ret;
}

##--------------------------------------------------------------
sub GenHeader {
  my($ret) = << "END_HEADER";
#include "nsVerifier.h"
END_HEADER

  return $ret;
}
##--------------------------------------------------------------
1; # this should be the last line
