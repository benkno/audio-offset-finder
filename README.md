audio-offset-finder
===================

This is a modified version of audio-offset-finder by BBC which has been
refactored to better handle batch processing. It also includes the significant
improvements from Abram Hindle's fork.

Installation
------------

    $ pip3 install --user git+https://github.com/benkno/audio-offset-finder.git

Usage
-----

    $ audio-offset-finder --help
    $ audio-offset-finder --find-offset-of file1.wav --within file2.wav
    Offset: 300 (seconds)

New FFMpeg features!

    @piggy:~/projects/audio-offset-finder$ python3 bin/audio-offset-finder --find-offset-of /media/hindle1/MyMedia3/Videos/20190629/001NOAH/MVI_4417.MOV  \
                                                                           --within /media/hindle1/MyMedia3/Videos/20190629/001NOAH/ZOOM0006.WAV \
                                                                           --ffmpeg
    Ref samples: 17654656 Find samples: 10964954
    ...
    Best matching window: 938
    Offset: 30.015782379212343 (seconds)
    Standard score: 20.952581258700146
    ffmpeg -i '/media/hindle1/MyMedia3/Videos/20190629/001NOAH/MVI_4417.MOV' -ss  '30.015782379212343' -i /media/hindle1/MyMedia3/Videos/20190629/001NOAH/ZOOM0006.WAV -map 0:v -map 1:a  -c copy -shortest '/media/hindle1/MyMedia3/Videos/20190629/001NOAH/MVI_4417.MOV.sync.mkv'
    ffmpeg version 4.1.4-0york3~18.04 Copyright (c) 2000-2019 the FFmpeg developers
      built with gcc 7 (Ubuntu 7.4.0-1ubuntu1~18.04.1)
    ...
    Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/media/hindle1/MyMedia3/Videos/20190629/001NOAH/MVI_4417.MOV':
      Metadata:
        major_brand     : qt  
        minor_version   : 537331968
        compatible_brands: qt  CAEP
        creation_time   : 2019-06-28T19:30:42.000000Z
      Duration: 00:11:25.31, start: 0.000000, bitrate: 46288 kb/s
        Stream #0:0(eng): Video: h264 (Constrained Baseline) (avc1 / 0x31637661), yuvj420p(pc, smpte170m/bt709/bt709), 1920x1080, 44750 kb/s, 23.98 fps, 23.98 tbr, 24k tbn, 48k tbc (default)
        Metadata:
          creation_time   : 2019-06-28T19:30:42.000000Z
        Stream #0:1(eng): Audio: pcm_s16le (sowt / 0x74776F73), 48000 Hz, stereo, s16, 1536 kb/s (default)
        Metadata:
          creation_time   : 2019-06-28T19:30:42.000000Z
    Guessed Channel Layout for Input Stream #1.0 : stereo
    Input #1, wav, from '/media/hindle1/MyMedia3/Videos/20190629/001NOAH/ZOOM0006.WAV':
      Metadata:
        encoded_by      : ZOOM Handy Recorder H1
        date            : 2010-06-21
        creation_time   : 19:50:32
        time_reference  : 6857472000
        coding_history  : A=PCM,F=96000,W=24,M=stereo,T=ZOOM Handy Recorder H1
      Duration: 00:18:23.42, bitrate: 4608 kb/s
        Stream #1:0: Audio: pcm_s24le ([1][0][0][0] / 0x0001), 96000 Hz, stereo, s32 (24 bit), 4608 kb/s
    File '/media/hindle1/MyMedia3/Videos/20190629/001NOAH/MVI_4417.MOV.sync.mkv' already exists. Overwrite ? [y/N] y
    Output #0, matroska, to '/media/hindle1/MyMedia3/Videos/20190629/001NOAH/MVI_4417.MOV.sync.mkv':
      Metadata:
        major_brand     : qt  
        minor_version   : 537331968
        compatible_brands: qt  CAEP
        encoder         : Lavf58.20.100
        Stream #0:0(eng): Video: h264 (Constrained Baseline) (avc1 / 0x31637661), yuvj420p(pc, smpte170m/bt709/bt709), 1920x1080, q=2-31, 44750 kb/s, 23.98 fps, 23.98 tbr, 1k tbn, 24k tbc (default)
        Metadata:
          creation_time   : 2019-06-28T19:30:42.000000Z
        Stream #0:1: Audio: pcm_s24le ([1][0][0][0] / 0x0001), 96000 Hz, stereo, s32 (24 bit), 4608 kb/s
    Stream mapping:
      Stream #0:0 -> #0:0 (copy)
      Stream #1:0 -> #0:1 (copy)
    Press [q] to stop, [?] for help
    frame=16431 fps=655 q=-1.0 Lsize= 4129923kB time=00:11:25.26 bitrate=49370.9kbits/s speed=27.3x    
    video:3743607kB audio:385467kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.020552%


Testing
-------

    $ nosetests

Currently broken 

Licensing terms and authorship
------------------------------

See 'COPYING' and 'AUTHORS' files.

The audio file used in the tests was downloaded from
[Wikimedia Commons](http://en.wikipedia.org/wiki/File:Tim_Berners-Lee_-_Today_-_9_July_2008.flac),
and was originally extracted from the 9 July 2008 
episode of the BBC [Today programme](http://www.bbc.co.uk/programmes/b00cddwc).
