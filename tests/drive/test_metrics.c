// Run from the repository root:
// gcc -I. -Iraylib-5.5_linux_amd64/include -ffunction-sections -fdata-sections \
//     tests/drive/test_metrics.c -Wl,--gc-sections -lm -o /tmp/test_drive_metrics
// /tmp/test_drive_metrics
#include <sys/wait.h>
#include "pufferlib/ocean/drive/drive.h"

static void test_outcomes(void) {
    for (int mode = GOAL_RESPAWN; mode <= GOAL_STOP; mode++) {
        Entity entities[8] = {0};
        Log logs[8] = {0};
        int indices[8];
        Drive env = {
            .entities = entities,
            .logs = logs,
            .active_agent_indices = indices,
            .active_agent_count = 8,
            .goal_behavior = mode,
        };
        // Cover all combinations of goal completion, collision, and off-road.
        for (int i = 0; i < 8; i++) {
            indices[i] = i;
            entities[i].goals_sampled_this_episode = 1;
            entities[i].goals_reached_this_episode = (i & 1) != 0;
            logs[i].collision_rate = (i & 2) != 0;
            logs[i].offroad_rate = (i & 4) != 0;
        }
        add_log(&env);
        assert(env.log.n == 8);
        assert(env.log.safe_completion_rate == 1);
        assert(env.log.safe_completion_rate / env.log.n == 0.125f);
    }
}

static void test_multiple_goals(void) {
    Entity entities[3] = {0};
    Log logs[3] = {0};
    int indices[] = {0, 1, 2};
    Drive env = {
        .entities = entities,
        .logs = logs,
        .active_agent_indices = indices,
        .active_agent_count = 3,
        .goal_behavior = GOAL_GENERATE_NEW,
    };
    entities[0].goals_sampled_this_episode = 3;
    entities[0].goals_reached_this_episode = 2; // Legacy score accepts this; strict success does not.
    entities[1].goals_sampled_this_episode = 3;
    entities[1].goals_reached_this_episode = 3;
    // Agent 2 has no assigned goals and must not count as a success.
    add_log(&env);
    assert(env.log.n == 3);
    assert(env.log.score == 2);
    assert(env.log.safe_completion_rate == 1);
}

static void test_respawn_preserves_violations(void) {
    for (int offroad = 0; offroad <= 1; offroad++) {
        float trajectory[] = {0};
        Entity entity = {
            .traj_x = trajectory, .traj_y = trajectory, .traj_heading = trajectory,
            .traj_vx = trajectory, .traj_vy = trajectory,
            .goals_sampled_this_episode = 1,
            .goals_reached_this_episode = 1,
            .collided_before_goal = 1,
        };
        Log log = {.collision_rate = !offroad, .offroad_rate = offroad};
        int index = 0;
        Drive env = {
            .entities = &entity,
            .logs = &log,
            .active_agent_indices = &index,
            .active_agent_count = 1,
            .goal_behavior = GOAL_RESPAWN,
        };
        respawn_agent(&env, 0);
        assert(entity.collided_before_goal == 0);
        add_log(&env);
        assert(env.log.score == 1); // Demonstrates why the old score is insufficient.
        assert(env.log.safe_completion_rate == 0);

        // Completing another goal after respawn must not erase the earlier failure.
        entity.goals_reached_this_episode++;
        add_log(&env);
        assert(env.log.safe_completion_rate == 0);
    }
}

static void test_violation_after_completion(void) {
    Entity entity = {.goals_sampled_this_episode = 1, .goals_reached_this_episode = 1};
    Log log = {0};
    int index = 0;
    Drive env = {
        .entities = &entity, .logs = &log, .active_agent_indices = &index,
        .active_agent_count = 1, .goal_behavior = GOAL_STOP,
    };
    add_log(&env);
    assert(env.log.safe_completion_rate == 1);
    env.log = (Log){0};
    log.collision_rate = 1;
    add_log(&env);
    assert(env.log.safe_completion_rate == 0);
    env.log = (Log){0};
    log.collision_rate = 0;
    log.offroad_rate = 1;
    add_log(&env);
    assert(env.log.safe_completion_rate == 0);
}

int main(void) {
    test_outcomes();
    test_multiple_goals();
    test_respawn_preserves_violations();
    test_violation_after_completion();
    puts("Drive safe completion metric tests passed.");
    return 0;
}
