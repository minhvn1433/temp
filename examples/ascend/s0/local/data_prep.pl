#!/usr/bin/perl

use open ':std', ':encoding(UTF-8)'; # Use UTF-8 encoding for standard streams

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-ASCEND-corpus> <dataset> <valid-train|valid-dev|valid-test>\n";
  exit(1);
}

# use ffmpeg
if (length(`which ffmpeg`) == 0) {
  print "Please install 'ffmpeg' on All worker nodes!\n";
  exit 1;
}


($db_base, $dataset, $out_dir) = @ARGV;
mkdir data unless -d data;
mkdir $out_dir unless -d $out_dir;


open(CSV, "<", "$db_base/$dataset.tsv") or die "cannot open dataset CSV file";
open(SPKR,">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(TEXT,">", "$out_dir/text") or die "Could not open the output file $out_dir/text";
open(WAV,">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";
my $header = <CSV>;
while(<CSV>) {
  chomp;
  my ($spkr, $filepath, $text, $accent) = split("\t", $_);
  $uttId = $filepath;
  if (-z "$filepath") {
    print "null file $filepath\n";
    next;
  }
  $uttId =~ s/\.wav//g;
  $uttId =~ tr/\//-/;
  # speaker information should be suffix of the utterance Id
  $uttId = "$spkr-$uttId";
  $text = uc($text);
  if (index($text, "{") != -1 and index($text, "}") != -1) {
      next;
  }

  print TEXT "$uttId"," ","$text","\n";
  print WAV "$uttId"," ","$filepath","\n";
  print SPKR "$uttId"," $spkr","\n";
}
close(SPKR) || die;
close(TEXT) || die;
close(WAV) || die;
close(CSV) || die;

if (system(
  "tools/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C tools/fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C tools/validate_data_dir.sh --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}