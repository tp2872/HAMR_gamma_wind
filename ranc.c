/* ranc -- return a random deviate between 0 and 1
   double ranc(iseed)
      * iseed = integer seed for random number generator
      - on first call, will seed with iseed (if > 0) or with time(0)
      - will reseed anytime iseed > 0
*/

#include <stdlib.h>
#include <time.h>

#define NRANC 64
#define RND	( 0x7fff & rand() )
#define MIN	( 2 << 8 )

    static int P[NRANC] = {
	46337, 46327, 46309, 46307, 46301, 46279, 46273, 46271,
	46261, 46237, 46229, 46219, 46199, 46187, 46183, 46181,
	46171, 46153, 46147, 46141, 46133, 46103, 46099, 46093,
	46091, 46073, 46061, 46051, 46049, 46027, 46021, 45989,
	45979, 45971, 45959, 45953, 45949, 45943, 45893, 45887,
	45869, 45863, 45853, 45841, 45833, 45827, 45823, 45821,
	45817, 45779, 45767, 45763, 45757, 45751, 45737, 45707,
	45697, 45691, 45677, 45673, 45667, 45659, 45641, 45631
	} ;

    static int a[NRANC] ;
    static int S[NRANC] ;
    static int n = 0 ;
    int called={1} ;

double ranc(int iseed)
{
    int i ;

/* seed the random number generator if first time called or if iseed > 0 */ 
    if (called || iseed != 0)  { 
	called = 0 ;
	if (iseed==0)
	    iseed = time(0) ;
	srand(iseed) ;
	n = 0 ;
	for( i = 0 ; i < NRANC ; i++ ) {
	    a[i] = 0.0 ;
	    S[i] = 0.0 ;
	    while( ( a[i] = RND ) < MIN || a[i] > P[i] - MIN ) ;
	    while( ( S[i] = RND ) <   1 || a[i] > P[i] -   1 ) ;
	}
    }

    n = S[n] & ( NRANC - 1 ) ;
    S[n] = ( S[n] * a[n] ) % P[n] ;
    return (double) S[n] / (double) P[n] ;
}

#undef NRANC 
#undef RND
#undef MIN
