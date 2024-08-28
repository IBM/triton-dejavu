
#include <unistd.h>
#include <signal.h>

void alarmHandler(int sig)
{
  alarm(0);
  //kill(getpid(), SIGTRAP);
  kill(getpid(), SIGKILL);
}


void setAlarmHandler(int timeout)
{
  signal(SIGALRM, alarmHandler);
  alarm(timeout);
}


