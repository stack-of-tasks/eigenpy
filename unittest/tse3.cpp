#include "libse3.hpp"



int main(int argc, char ** argv)
{
  buildFsMap(argc==1);

  int nbLoop=100;
  int nbTest=argc-1;

  Eigen::MatrixXd times(nbTest,nbLoop);

  for(int loop=0;loop<nbLoop;++loop)
    for(int i=1;i<argc;++i)
      {
	std::string name = argv[i];
	times(i-1,loop) = fs[name].first();
      }

  std::cout << "Times = \n" << times << std::endl << std::endl << std::endl;
  for(int i=1;i<argc;++i)
    {
      std::string name = argv[i];
      std::cout << times.row(i-1).mean() << "  \t==>\t    " << fs[name].second << std::endl;
    }

  return 0;
}
 
