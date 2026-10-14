#pragma once
#include <InputManager.hpp>
#include <algorithm>

namespace SokuRLBridge {
// ReDelay/SokuAI updateInput convention: one update per simulated frame.
inline void advanceInputCounters(SokuLib::KeyInput &old, const SokuLib::KeyInput &held) {
    auto button=[](int previous,int down) { return down ? std::min(previous,1000000)+1 : 0; };
    old.a=button(old.a,held.a); old.b=button(old.b,held.b);
    old.c=button(old.c,held.c); old.d=button(old.d,held.d);
    old.changeCard=button(old.changeCard,held.changeCard);
    old.spellcard=button(old.spellcard,held.spellcard);
    auto axis=[](int previous,int direction) {
        return direction>0 ? std::min(std::max(0,previous),1000000)+1 :
            direction<0 ? std::max(std::min(0,previous),-1000000)-1 : 0;
    };
    old.horizontalAxis=axis(old.horizontalAxis,held.horizontalAxis);
    old.verticalAxis=axis(old.verticalAxis,held.verticalAxis);
}
}
