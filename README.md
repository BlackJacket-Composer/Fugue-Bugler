# Fugue-Bugler
Musical counterpoint belief propagation - Final project for G22-2565 Machine Learning, Fall 2011

Does one hand ever know what the other is doing? In musical theory, it must! This is an implementation of
Generalized Belief Propagation for musical counterpoint notes. Given a training set of musical data in a
particular style (i.e. a nursery rhyme or Bach), can we predict a single note's counterpoint? How about 
generate a completely new sequence, based on a each note's likelyhood?

Let's take the sequence C-5, B-4, A-4, B-4, C-5, D-5. Can we accurately predict, given enough training
information, the Alto counterpoint to these Soprano notes? If we're given five of the counterpoints as
E-4, G-4, F-4, ?, E-4, B-4, we should be able to determine that the missing note is G-4 using the shape
of the preceding and following notes, coupled the rules we know about counterpoint species.

We implemented belief propagation on the first species only. Its rules are:
1. Begin and end on either the unison, octave, or fifth, unless the added part is underneath, in which case begin and end only on unison or octave.
2. Use no unisons except at the beginning or end.
3. Avoid parallel fifths or octaves between any two parts; and avoid "hidden" parallel fifths or octaves: that is, movement by similar motion to a perfect fifth or octave, unless one part (sometimes restricted to the higher of the parts) moves by step.
4. Avoid moving in parallel fourths. (In practice Palestrina and others frequently allowed themselves such progressions, especially if they do not involve the lowest of the parts.)
5. Avoid moving in parallel thirds or sixths for very long.
6. Attempt to keep any two adjacent parts within a tenth of each other, unless an exceptionally pleasing line can be written by moving outside of that range.
7. Avoid having any two parts move in the same direction by skip.
8. Attempt to have as much contrary motion as possible.
9. Avoid dissonant intervals between any two parts: major or minor 2nd, major or minor 7th, any augmented or diminished interval, and perfect fourth (in many contexts).

The belief propagation algorithm is based on http://ssg.mit.edu/nbp

Our project forks Anthony Theocharis' excellent Counterpoint.py library as a basis for reading in midi notes.
Improvements suggested upstream.

To run:

    Usage: counterpoint.py [options]
    
    Options:
      -h, --help            show this help message and exit
     
       -t                    Read tracks from tracks.py and perform sparse-
     						sample belief prop on missing notes. If encountered,
     						will ignore instructions to read from MIDI file.
     
       -f, --fill-in			Given a full length, single-voice piece generate an
     						original counterpoint
     
       -g, --guess			Fill in missing notes from the MIDI file. Used in
     						combination with like options
