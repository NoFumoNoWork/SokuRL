#pragma once

namespace SokuRLBridge {
bool installCrashReport();
void closeCrashReport();
void traceResetStage(const char *stage, const void *scene);
}
