#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>
#include <pthread.h>

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

// Trajectory Length
#define TRAJECTORY_LENGTH 91

// Actions
#define NOOP 0

// Dynamics Models
#define CLASSIC 0
#define INVERTIBLE_BICYLE 1
#define DELTA_LOCAL 2
#define STATE_DYNAMICS 3

// collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// grid cell size
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 10
#define SLOTS_PER_CELL (MAX_ENTITIES_PER_CELL * 2 + 1)

// Max road segment observation entities
#define MAX_ROAD_SEGMENT_OBSERVATIONS 200
#define MAX_CARS 64
// Observation Space Constants
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MIN_REL_GOAL_COORD -1000.0f
#define MAX_REL_GOAL_COORD 1000.0f
#define MIN_REL_AGENT_POS -1000.0f
#define MAX_REL_AGENT_POS 1000.0f
#define MAX_ORIENTATION_RAD 2 * PI
#define MIN_RG_COORD -1000.0f
#define MAX_RG_COORD 1000.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f
#define DIM_EGO 6
#define DIM_PARTNER 7
#define DIM_ROAD 7
#define MAX_ACCELERATION 4.0f
#define MAX_STEERING_ANGLE 0.6f
#define MIN_ACCELERATION -4.0f
#define MIN_STEERING_ANGLE -0.6f

#define DISCOUNT_FACTOR 0.9f
#define RADIUS_GOAL 2.0f

// Acceleration Values
static const float ACCELERATION_VALUES[9] = {-4.0000f, -3.0000f, -2.0000f, -1.0000f, 0.0000f, 1.0000f, 2.0000f, 3.0000f, 4.0000f};
static const float STEERING_VALUES[13] = {-0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

static const float offsets[4][2] = {
    {-1, 1}, // top-left
    {1, 1},  // top-right
    {1, -1}, // bottom-right
    {-1, -1} // bottom-left
};

#define TRAJ_VOCAB_SIZE 16384
#define TRAJ_VOCAB_LEN 5
#define TRAJ_VOCAB_DIM 3
#define T3_INDEX(t,i,j,k) ( ((size_t)(i) * (t)->n2 + (size_t)(j)) * (t)->n3 + (size_t)(k) )
#define T3_GET(t,i,j,k)   ( (t)->data[T3_INDEX((t),(i),(j),(k))] )
#define T3_SET(t,i,j,k,v) ( (t)->data[T3_INDEX((t),(i),(j),(k))] = (v) )

#define DEBUG 0

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, // Top row
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, // Second row
    {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, // Middle row (including center)
    {-2, 1}, {-1, 1}, {0, 1}, {1, 1}, {2, 1}, // Fourth row
    {-2, 2}, {-1, 2}, {0, 2}, {1, 2}, {2, 2} // Bottom row
};

struct timespec ts;

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;

struct Log
{
    float episode_return;
    float episode_length;
    float perf;
    float score;
    float offroad_rate;
    float collision_rate;
    float clean_collision_rate;
    float completion_rate;
    float dnf_rate;
    float n;
};

typedef struct Tensor3f Tensor3f;
struct Tensor3f
{
    size_t n1, n2, n3;   // dims
    float *data;         // flat buffer, length = n1*n2*n3
};

typedef struct Entity Entity;
struct Entity
{
    int type;
    int array_size;
    float *traj_x;
    float *traj_y;
    float *traj_z;
    float *traj_vx;
    float *traj_vy;
    float *traj_vz;
    float *traj_heading;
    int *traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    int mark_as_expert;
    int collision_state;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    int valid;
    int reached_goal;
    int reached_goal_in_horizon;
    int respawn_timestep;
    int collided_before_goal;
    int reached_goal_this_episode;
    int active_agent;

    struct {
        int   collision_state;
        float x, y, z;
        float vx, vy, vz;
        float heading;
        float heading_x;
        float heading_y;
        int   valid;
        int   reached_goal;
        int   respawn_timestep;
        int   collided_before_goal;
        int   reached_goal_this_episode;
        int   active_agent;
    } backup;
};

static inline void entity_save(Entity *e)
{
    e->backup.collision_state       = e->collision_state;
    e->backup.x = e->x;  e->backup.y = e->y;  e->backup.z = e->z;
    e->backup.vx = e->vx; e->backup.vy = e->vy; e->backup.vz = e->vz;
    e->backup.heading   = e->heading;
    e->backup.heading_x = e->heading_x;
    e->backup.heading_y = e->heading_y;
    e->backup.valid     = e->valid;
    e->backup.reached_goal = e->reached_goal;
    e->backup.respawn_timestep = e->respawn_timestep;
    e->backup.collided_before_goal = e->collided_before_goal;
    e->backup.reached_goal_this_episode = e->reached_goal_this_episode;
    e->backup.active_agent = e->active_agent;
}

static inline void entity_restore(Entity *e)
{
    e->collision_state = e->backup.collision_state;
    e->x = e->backup.x;  e->y = e->backup.y;  e->z = e->backup.z;
    e->vx = e->backup.vx; e->vy = e->backup.vy; e->vz = e->backup.vz;
    e->heading   = e->backup.heading;
    e->heading_x = e->backup.heading_x;
    e->heading_y = e->backup.heading_y;
    e->valid     = e->backup.valid;
    e->reached_goal = e->backup.reached_goal;
    e->respawn_timestep = e->backup.respawn_timestep;
    e->collided_before_goal = e->backup.collided_before_goal;
    e->reached_goal_this_episode = e->backup.reached_goal_this_episode;
    e->active_agent = e->backup.active_agent;
}

void free_entity(Entity *entity)
{
    // free trajectory arrays
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
}

float relative_distance(float a, float b)
{
    float distance = sqrtf(powf(a - b, 2));
    return distance;
}

float relative_distance_2d(float x1, float y1, float x2, float y2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx * dx + dy * dy);
    return distance;
}

struct Drive
{
    Client *client;
    float *observations;
    int *actions;
    float *rewards;
    unsigned char *terminals;
    Log log;
    Log *logs;
    int num_agents;
    int active_agent_count;
    int *active_agent_indices;
    int human_agent_idx;
    Entity *entities;
    int num_entities;
    int num_cars;
    int num_objects;
    int num_roads;
    int static_car_count;
    int *static_car_indices;
    int expert_static_car_count;
    int *expert_static_car_indices;
    int timestep;
    int dynamics_model;
    float *map_corners;
    int *grid_cells; // holds entity ids and geometry index per cell
    int grid_cols;
    int grid_rows;
    int vision_range;
    int *neighbor_offsets;
    int *neighbor_cache_entities;
    int *neighbor_cache_indices;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    char *map_name;
    float world_mean_x;
    float world_mean_y;
    int spawn_immunity_timer;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
};

void add_log(Drive *env)
{
    for (int i = 0; i < env->active_agent_count; i++)
    {
        Entity *e = &env->entities[env->active_agent_indices[i]];
        if (e->reached_goal_this_episode)
            env->log.completion_rate += 1.0f;
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        int clean_collided = env->logs[i].clean_collision_rate;
        env->log.clean_collision_rate += clean_collided;
        if (e->reached_goal_this_episode && !e->collided_before_goal)
        {
            env->log.score += 1.0f;
            env->log.perf += 1.0f;
        }
        if (!offroad && !collided && !e->reached_goal_this_episode)
            env->log.dnf_rate += 1.0f;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.n += 1;
    }
}

static Tensor3f traj_vocab = {0,0,0,NULL};
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int traj_vocab_init_status = -1;   // 0 = ok, nonzero = error code

static int t3_load_raw(const char *filename,
                       size_t n1, size_t n2, size_t n3,
                       Tensor3f *out)
{
    size_t total = n1 * n2 * n3;
    FILE *f = fopen(filename, "rb");
    float *buf;
    size_t nread;

    if (!f) { perror("fopen"); return 2; }

    buf = (float*)malloc(total * sizeof(*buf));
    if (!buf) { perror("malloc"); fclose(f); return 1; }

    nread = fread(buf, sizeof(float), total, f);
    fclose(f);

    if (nread != total) {
        fprintf(stderr, "Expected %zu floats, read %zu\n", total, nread);
        free(buf);
        return 3;
    }

    out->n1 = n1; out->n2 = n2; out->n3 = n3;
    out->data = buf;
    return 0;
}

// static void traj_vocab_cleanup(void)
// {
//     if (traj_vocab.data) {
//         free(traj_vocab.data);
//         traj_vocab.data = NULL;
//     }
//     traj_vocab.n1 = traj_vocab.n2 = traj_vocab.n3 = 0;
// }

// static void traj_vocab_init_once(void)
// {
//     /* Runs exactly once across all threads */
//     traj_vocab_init_status = t3_load_raw("/datasets_local/yyin5/gpudrive_original/data/binaries/sorted_smoothed_traj_vocab_16384.bin", TRAJ_VOCAB_SIZE, TRAJ_VOCAB_LEN, TRAJ_VOCAB_DIM, &traj_vocab);

//     /* Optional: auto-cleanup at process exit */
//     if (traj_vocab_init_status == 0) {
//         atexit(traj_vocab_cleanup);
//     }
// }

// /* Public API: ensure tensor is loaded once; return 0 on success */
// int ensure_tensor_loaded(void)
// {
//     pthread_once(&g_once, traj_vocab_init_once);
//     return traj_vocab_init_status;
// }

Entity *load_map_binary(const char *filename, Drive *env)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
        return NULL;
    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;
    Entity *entities = (Entity *)malloc(env->num_entities * sizeof(Entity));
    for (int i = 0; i < env->num_entities; i++)
    {
        // Read base entity data
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float *)malloc(size * sizeof(float));
        entities[i].traj_y = (float *)malloc(size * sizeof(float));
        entities[i].traj_z = (float *)malloc(size * sizeof(float));
        if (entities[i].type == 1 || entities[i].type == 2 || entities[i].type == 3)
        { // Object type
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float *)malloc(size * sizeof(float));
            entities[i].traj_vy = (float *)malloc(size * sizeof(float));
            entities[i].traj_vz = (float *)malloc(size * sizeof(float));
            entities[i].traj_heading = (float *)malloc(size * sizeof(float));
            entities[i].traj_valid = (int *)malloc(size * sizeof(int));
        }
        else
        {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (entities[i].type == 1 || entities[i].type == 2 || entities[i].type == 3)
        { // Object type
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
        }
        // Read remaining scalar fields
        fread(&entities[i].width, sizeof(float), 1, file);
        fread(&entities[i].length, sizeof(float), 1, file);
        fread(&entities[i].height, sizeof(float), 1, file);
        fread(&entities[i].goal_position_x, sizeof(float), 1, file);
        fread(&entities[i].goal_position_y, sizeof(float), 1, file);
        fread(&entities[i].goal_position_z, sizeof(float), 1, file);
        fread(&entities[i].mark_as_expert, sizeof(int), 1, file);
    }
    fclose(file);
    return entities;
}

void set_start_position(Drive *env)
{
    // InitWindow(800, 600, "GPU Drive");
    // BeginDrawing();
    for (int i = 0; i < env->num_entities; i++)
    {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++)
        {
            if (env->active_agent_indices[j] == i)
            {
                is_active = 1;
                break;
            }
        }
        Entity *e = &env->entities[i];
        e->x = e->traj_x[0];
        e->y = e->traj_y[0];
        e->z = e->traj_z[0];

        if (e->type > 3 || e->type == 0)
            continue;

        if (is_active == 0)
        {
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->reached_goal = 0;
            e->reached_goal_in_horizon = 0;
            e->collided_before_goal = 0;
        }
        else
        {
            e->vx = e->traj_vx[0];
            e->vy = e->traj_vy[0];
            e->vz = e->traj_vz[0];
        }
        e->heading = e->traj_heading[0];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->valid = e->traj_valid[0];
        e->collision_state = 0;
        e->respawn_timestep = -1;
    }
    // EndDrawing();
    int x = 0;
}

int get_grid_index(Drive *env, float x1, float y1)
{
    if (env->map_corners[0] >= env->map_corners[2] || env->map_corners[1] >= env->map_corners[3])
    {
        printf("Invalid grid coordinates\n");
        return -1; // Invalid grid coordinates
    }
    float world_width = env->map_corners[2] - env->map_corners[0];  // Positive value
    float world_height = env->map_corners[3] - env->map_corners[1]; // Positive value
    int cells_x = (int)ceil(world_width / GRID_CELL_SIZE);          // Number of columns
    int cells_y = (int)ceil(world_height / GRID_CELL_SIZE);         // Number of rows
    float relative_x = x1 - env->map_corners[0];                    // Distance from left
    float relative_y = y1 - env->map_corners[1];                    // Distance from top
    int grid_x = (int)(relative_x / GRID_CELL_SIZE);                // Column index
    int grid_y = (int)(relative_y / GRID_CELL_SIZE);                // Row index
    if (grid_x < 0 || grid_x >= cells_x || grid_y < 0 || grid_y >= cells_y)
        return -1; // Return -1 for out of bounds
    int index = (grid_y * cells_x) + grid_x;
    return index;
}

void add_entity_to_grid(Drive *env, int grid_index, int entity_idx, int geometry_idx)
{
    if (grid_index == -1)
    {
        return;
    }
    int base_index = grid_index * SLOTS_PER_CELL;
    int count = env->grid_cells[base_index];
    if (count >= MAX_ENTITIES_PER_CELL)
        return;
    env->grid_cells[base_index + count * 2 + 1] = entity_idx;
    env->grid_cells[base_index + count * 2 + 2] = geometry_idx;
    env->grid_cells[base_index] = count + 1;
}

void init_grid_map(Drive *env)
{
    // Find top left and bottom right points of the map
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int first_valid_point = 0;
    for (int i = 0; i < env->num_entities; i++)
    {
        if (env->entities[i].type > 3 && env->entities[i].type < 7)
        {
            // Check all points in the trajectory for road elements
            Entity *e = &env->entities[i];
            for (int j = 0; j < e->array_size; j++)
            {
                if (e->traj_x[j] == -10000)
                    continue;
                if (e->traj_y[j] == -10000)
                    continue;
                if (!first_valid_point)
                {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = true;
                    continue;
                }
                if (e->traj_x[j] < top_left_x)
                    top_left_x = e->traj_x[j];
                if (e->traj_x[j] > bottom_right_x)
                    bottom_right_x = e->traj_x[j];
                if (e->traj_y[j] < top_left_y)
                    top_left_y = e->traj_y[j];
                if (e->traj_y[j] > bottom_right_y)
                    bottom_right_y = e->traj_y[j];
            }
        }
    }

    env->map_corners = (float *)calloc(4, sizeof(float));
    env->map_corners[0] = top_left_x;
    env->map_corners[1] = top_left_y;
    env->map_corners[2] = bottom_right_x;
    env->map_corners[3] = bottom_right_y;

    // Calculate grid dimensions
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = bottom_right_y - top_left_y;
    env->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_cols * env->grid_rows;
    env->grid_cells = (int *)calloc(grid_cell_count * SLOTS_PER_CELL, sizeof(int));
    // Populate grid cells
    for (int i = 0; i < env->num_entities; i++)
    {
        if (env->entities[i].type > 3 && env->entities[i].type < 7)
        {
            for (int j = 0; j < env->entities[i].array_size - 1; j++)
            {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = get_grid_index(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j);
            }
        }
    }
}

void init_neighbor_offsets(Drive *env)
{
    // Allocate memory for the offsets
    env->neighbor_offsets = (int *)calloc(env->vision_range * env->vision_range * 2, sizeof(int));
    // neighbor offsets in a spiral pattern
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0;                  // Current x offset
    int y = 0;                  // Current y offset
    int dir = 0;                // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1;      // Number of steps in current direction
    int steps_taken = 0;        // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0;              // Total offsets added
    int max_offsets = env->vision_range * env->vision_range;
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0; // x offset
    env->neighbor_offsets[curr_idx++] = 0; // y offset
    total++;
    // Generate spiral pattern
    while (total < max_offsets)
    {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        // Only add if within vision range bounds
        if (abs(x) <= env->vision_range / 2 && abs(y) <= env->vision_range / 2)
        {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        // Check if we need to change direction
        if (steps_taken != steps_to_take)
            continue;
        steps_taken = 0;     // Reset steps taken
        dir = (dir + 1) % 4; // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        // Increase step length every two direction changes
        if (segments_completed % 2 == 0)
            steps_to_take++;
    }
}

void cache_neighbor_offsets(Drive *env)
{
    int count = 0;
    int cell_count = env->grid_cols * env->grid_rows;
    for (int i = 0; i < cell_count; i++)
    {
        int cell_x = i % env->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_cols;
        env->neighbor_cache_indices[i] = count;
        for (int j = 0; j < env->vision_range * env->vision_range; j++)
        {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_cols * y + x;
            if (x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows)
                continue;
            int grid_count = env->grid_cells[grid_index * SLOTS_PER_CELL];
            count += grid_count * 2;
        }
    }
    env->neighbor_cache_indices[cell_count] = count;
    env->neighbor_cache_entities = (int *)calloc(count, sizeof(int));
    for (int i = 0; i < cell_count; i++)
    {
        int neighbor_cache_base_index = 0;
        int cell_x = i % env->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_cols;
        for (int j = 0; j < env->vision_range * env->vision_range; j++)
        {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_cols * y + x;
            if (x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows)
                continue;
            int grid_count = env->grid_cells[grid_index * SLOTS_PER_CELL];
            int base_index = env->neighbor_cache_indices[i];
            int src_idx = grid_index * SLOTS_PER_CELL + 1;
            int dst_idx = base_index + neighbor_cache_base_index;
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(&env->neighbor_cache_entities[dst_idx],
                   &env->grid_cells[src_idx],
                   grid_count * 2 * sizeof(int));

            // Update index outside the loop
            neighbor_cache_base_index += grid_count * 2;
        }
    }
}

int get_neighbor_cache_entities(Drive *env, int cell_idx, int *entities, int max_entities)
{
    if (cell_idx < 0 || cell_idx >= (env->grid_cols * env->grid_rows))
        return 0; // Invalid cell index
    int base_index = env->neighbor_cache_indices[cell_idx];
    int end_index = env->neighbor_cache_indices[cell_idx + 1];
    int count = end_index - base_index;
    int pairs = count / 2; // Entity ID and geometry ID pairs
    // Limit to available space
    if (pairs > max_entities)
    {
        pairs = max_entities;
        count = pairs * 2;
    }
    memcpy(entities, env->neighbor_cache_entities + base_index, count * sizeof(int));
    return pairs;
}

void set_means(Drive *env)
{
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    // Compute single mean for all entities (vehicles and roads)
    for (int i = 0; i < env->num_entities; i++)
    {
        if (env->entities[i].type == VEHICLE)
        {
            for (int j = 0; j < env->entities[i].array_size; j++)
            {
                // Assume a validity flag exists (e.g., valid[j]); adjust if not available
                if (env->entities[i].traj_valid[j])
                { // Add validity check if applicable
                    point_count++;
                    mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                    mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
                }
            }
        }
        else if (env->entities[i].type >= 4)
        {
            for (int j = 0; j < env->entities[i].array_size; j++)
            {
                point_count++;
                mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
            }
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;
    for (int i = 0; i < env->num_entities; i++)
    {
        if (env->entities[i].type == VEHICLE || env->entities[i].type >= 4)
        {
            for (int j = 0; j < env->entities[i].array_size; j++)
            {
                if (env->entities[i].traj_x[j] == -10000)
                    continue;
                env->entities[i].traj_x[j] -= mean_x;
                env->entities[i].traj_y[j] -= mean_y;
            }
            env->entities[i].goal_position_x -= mean_x;
            env->entities[i].goal_position_y -= mean_y;
        }
    }
}

void move_expert(Drive *env, int agent_idx, int timestep)
{
    Entity *agent = &env->entities[agent_idx];
    agent->x = agent->traj_x[timestep];
    agent->y = agent->traj_y[timestep];
    agent->z = agent->traj_z[timestep];
    agent->heading = agent->traj_heading[timestep];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
}

bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2])
{
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
        fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
        return false;

    // Calculate vectors
    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];

    // Calculate cross products
    float cross = dx1 * dy2 - dy1 * dx2;

    // If lines are parallel
    if (cross == 0)
        return false;

    // Calculate relative vectors between start points
    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];

    // Calculate parameters for intersection point
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;

    // Check if intersection point lies within both line segments
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int check_neighbors(Drive *env, float x, float y, int *entity_list, int max_size, const int (*local_offsets)[2], int offset_size)
{
    // Get the grid index for the given position (x, y)
    int index = get_grid_index(env, x, y);
    if (index == -1)
        return 0; // Return 0 size if position invalid
    // Calculate 2D grid coordinates
    int cells_x = env->grid_cols;
    int grid_x = index % cells_x;
    int grid_y = index / cells_x;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++)
    {
        int nx = grid_x + local_offsets[i][0];
        int ny = grid_y + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx < 0 || nx >= env->grid_cols || ny < 0 || ny >= env->grid_rows)
            continue;
        int neighbor_idx = (ny * env->grid_cols + nx) * SLOTS_PER_CELL;
        int count = env->grid_cells[neighbor_idx];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++)
        {
            int entity_id = env->grid_cells[neighbor_idx + 1 + j * 2];
            int geometry_idx = env->grid_cells[neighbor_idx + 2 + j * 2];
            entity_list[entity_list_count] = entity_id;
            entity_list[entity_list_count + 1] = geometry_idx;
            entity_list_count += 2;
        }
    }
    return entity_list_count;
}

int check_aabb_collision(Entity *car1, Entity *car2)
{
    // Get car corners in world space
    float cos1 = car1->heading_x;
    float sin1 = car1->heading_y;
    float cos2 = car2->heading_x;
    float sin2 = car2->heading_y;

    // Calculate half dimensions
    float half_len1 = car1->length * 0.5f;
    float half_width1 = car1->width * 0.5f;
    float half_len2 = car2->length * 0.5f;
    float half_width2 = car2->width * 0.5f;

    // Calculate car1's corners in world space
    float car1_corners[4][2] = {
        {car1->x + (half_len1 * cos1 - half_width1 * sin1), car1->y + (half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (half_len1 * cos1 + half_width1 * sin1), car1->y + (half_len1 * sin1 - half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 - half_width1 * sin1), car1->y + (-half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 + half_width1 * sin1), car1->y + (-half_len1 * sin1 - half_width1 * cos1)}};

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->x + (half_len2 * cos2 - half_width2 * sin2), car2->y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (half_len2 * cos2 + half_width2 * sin2), car2->y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 - half_width2 * sin2), car2->y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 + half_width2 * sin2), car2->y + (-half_len2 * sin2 - half_width2 * cos2)}};

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {cos1, sin1},  // Car1's length axis
        {-sin1, cos1}, // Car1's width axis
        {cos2, sin2},  // Car2's length axis
        {-sin2, cos2}  // Car2's width axis
    };

    // Check each axis
    for (int i = 0; i < 4; i++)
    {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;

        // Project car1's corners onto the axis
        for (int j = 0; j < 4; j++)
        {
            float proj = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj);
            max1 = fmaxf(max1, proj);
        }

        // Project car2's corners onto the axis
        for (int j = 0; j < 4; j++)
        {
            float proj = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj);
            max2 = fmaxf(max2, proj);
        }

        // If there's a gap on this axis, the boxes don't intersect
        if (max1 < min2 || min1 > max2)
            return 0; // No collision
    }

    // If we get here, there's no separating axis, so the boxes intersect
    return 1; // Collision
}

int collision_check(Drive *env, int agent_idx)
{
    Entity *agent = &env->entities[agent_idx];
    if (agent->x == -10000.0f)
        return -1;
    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float corners[4][2];
    for (int i = 0; i < 4; i++)
    {
        corners[i][0] = agent->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] = agent->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }
    int collided = 0;
    int car_collided_with_index = -1;
    int entity_list[MAX_ENTITIES_PER_CELL * 2 * 25]; // Array big enough for all neighboring cells
    int list_size = check_neighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL * 2 * 25, collision_offsets, 25);
    for (int i = 0; i < list_size; i += 2)
    {
        if (entity_list[i] == -1)
            continue;
        if (entity_list[i] == agent_idx)
            continue;
        Entity *entity;
        entity = &env->entities[entity_list[i]];
        if (entity->type != ROAD_EDGE)
            continue;
        int geometry_idx = entity_list[i + 1];
        float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
        float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
        for (int k = 0; k < 4; k++)
        { // Check each edge of the bounding box
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], start, end))
            {
                collided = OFFROAD;
                break;
            }
        }
        if (collided == OFFROAD)
            break;
    }
    for (int i = 0; i < MAX_CARS; i++)
    {
        int index = -1;
        if (i < env->active_agent_count)
            index = env->active_agent_indices[i];
        else if (i < env->num_cars)
            index = env->static_car_indices[i - env->active_agent_count];

        if (index == -1)
            continue;
        if (index == agent_idx)
            continue;
        Entity *entity = &env->entities[index];
        float x1 = entity->x;
        float y1 = entity->y;
        float dist = ((x1 - agent->x) * (x1 - agent->x) + (y1 - agent->y) * (y1 - agent->y));
        if (dist > 225.0f)
            continue;
        if (check_aabb_collision(agent, entity))
        {
            collided = VEHICLE_COLLISION;
            car_collided_with_index = index;
            break;
        }
    }
    agent->collision_state = collided;
    // spawn immunity for collisions with other agent cars as agent_idx respawns
    int is_active_agent = env->entities[agent_idx].active_agent;
    int respawned = env->entities[agent_idx].respawn_timestep != -1;
    int exceeded_spawn_immunity_agent = (env->timestep - env->entities[agent_idx].respawn_timestep) >= env->spawn_immunity_timer;
    if (collided == VEHICLE_COLLISION && is_active_agent == 1 && respawned)
        agent->collision_state = 0;

    // spawn immunity for collisions with other cars who just respawned
    if (collided == OFFROAD)
        return -1;
    if (car_collided_with_index == -1)
        return -1;

    int respawned_collided_with_car = env->entities[car_collided_with_index].respawn_timestep != -1;
    int exceeded_spawn_immunity_collided_with_car = (env->timestep - env->entities[car_collided_with_index].respawn_timestep) >= env->spawn_immunity_timer;
    int within_spawn_immunity_collided_with_car = (env->timestep - env->entities[car_collided_with_index].respawn_timestep) < env->spawn_immunity_timer;
    if (respawned_collided_with_car)
        agent->collision_state = 0;

    return car_collided_with_index;
}

int valid_active_agent(Drive *env, int agent_idx)
{
    float cos_heading = cosf(env->entities[agent_idx].traj_heading[0]);
    float sin_heading = sinf(env->entities[agent_idx].traj_heading[0]);
    float goal_x = env->entities[agent_idx].goal_position_x - env->entities[agent_idx].traj_x[0];
    float goal_y = env->entities[agent_idx].goal_position_y - env->entities[agent_idx].traj_y[0];
    // Rotate to ego vehicle's frame
    float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
    float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, rel_goal_x, rel_goal_y);
    env->entities[agent_idx].width *= 0.7f;
    env->entities[agent_idx].length *= 0.7f;
    if (distance_to_goal >= 2.0f && env->entities[agent_idx].mark_as_expert == 0 && env->active_agent_count < env->num_agents)
        return distance_to_goal;
    return 0;
}

void set_active_agents(Drive *env)
{
    env->active_agent_count = 0;
    env->static_car_count = 0;
    env->num_cars = 1;
    env->expert_static_car_count = 0;
    int active_agent_indices[MAX_CARS];
    int static_car_indices[MAX_CARS];
    int expert_static_car_indices[MAX_CARS];

    if (env->num_agents == 0)
        env->num_agents = MAX_CARS;
    int first_agent_id = env->num_objects - 1;
    float distance_to_goal = valid_active_agent(env, first_agent_id);
    if (distance_to_goal)
    {
        env->active_agent_count = 1;
        active_agent_indices[0] = first_agent_id;
        env->entities[first_agent_id].active_agent = 1;
        env->num_cars = 1;
    }
    else
    {
        env->active_agent_count = 0;
        env->num_cars = 0;
    }
    for (int i = 0; i < env->num_objects - 1 && env->num_cars < MAX_CARS; i++)
    {
        if (env->entities[i].type != 1)
            continue;
        if (env->entities[i].traj_valid[0] != 1)
            continue;
        env->num_cars++;
        float distance_to_goal = valid_active_agent(env, i);
        if (distance_to_goal > 0)
        {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
        }
        else
        {
            static_car_indices[env->static_car_count] = i;
            env->static_car_count++;
            env->entities[i].active_agent = 0;
            if (env->entities[i].mark_as_expert == 1 || (distance_to_goal >= 2.0f && env->active_agent_count == env->num_agents))
            {
                expert_static_car_indices[env->expert_static_car_count] = i;
                env->expert_static_car_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }
    // set up initial active agents
    env->active_agent_indices = (int *)malloc(env->active_agent_count * sizeof(int));
    env->static_car_indices = (int *)malloc(env->static_car_count * sizeof(int));
    env->expert_static_car_indices = (int *)malloc(env->expert_static_car_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; i++)
        env->active_agent_indices[i] = active_agent_indices[i];
    for (int i = 0; i < env->static_car_count; i++)
        env->static_car_indices[i] = static_car_indices[i];
    for (int i = 0; i < env->expert_static_car_count; i++)
        env->expert_static_car_indices[i] = expert_static_car_indices[i];
}

void remove_bad_trajectories(Drive *env)
{
    set_start_position(env);
    int legal_agent_count = 0;
    int legal_trajectories[env->active_agent_count];
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));
    // move experts through trajectories to check for collisions and remove as illegal agents
    for (int t = 0; t < TRAJECTORY_LENGTH; t++)
    {
        for (int i = 0; i < env->active_agent_count; i++)
        {
            int agent_idx = env->active_agent_indices[i];
            move_expert(env, agent_idx, env->timestep);
        }
        for (int i = 0; i < env->expert_static_car_count; i++)
        {
            int expert_idx = env->expert_static_car_indices[i];
            if (env->entities[expert_idx].x == -10000)
                continue;
            move_expert(env, expert_idx, env->timestep);
        }
        // check collisions
        for (int i = 0; i < env->active_agent_count; i++)
        {
            int agent_idx = env->active_agent_indices[i];
            env->entities[agent_idx].collision_state = 0;
            int collided_with_index = collision_check(env, agent_idx);
            if (env->entities[agent_idx].collision_state > 0 && collided_agents[i] == 0)
            {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with_index;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++)
    {
        if (collided_with_indices[i] == -1)
            continue;
        for (int j = 0; j < env->static_car_count; j++)
        {
            int static_car_idx = env->static_car_indices[j];
            if (static_car_idx != collided_with_indices[i])
                continue;
            env->entities[static_car_idx].traj_x[0] = -10000;
            env->entities[static_car_idx].traj_y[0] = -10000;
        }
    }
    env->timestep = 0;
}

void init(Drive *env)
{
    // if (ensure_tensor_loaded() != 0) {
    //     fprintf(stderr, "Failed to load %s (status=%d)\n",
    //             "/datasets_local/yyin5/gpudrive_original/data/binaries/traj_vocab_16384.bin", traj_vocab_init_status);
    //     return 1;
    // }
    env->human_agent_idx = 0;
    env->timestep = 0;
    env->entities = load_map_binary(env->map_name, env);
    env->dynamics_model = CLASSIC;
    set_means(env);
    init_grid_map(env);
    env->vision_range = 21;
    init_neighbor_offsets(env);
    env->neighbor_cache_indices = (int *)calloc((env->grid_cols * env->grid_rows) + 1, sizeof(int));
    cache_neighbor_offsets(env);
    set_active_agents(env);
    remove_bad_trajectories(env);
    set_start_position(env);
    env->logs = (Log *)calloc(env->active_agent_count, sizeof(Log));
}

void c_close(Drive *env)
{
    for (int i = 0; i < env->num_entities; i++)
        free_entity(&env->entities[i]);
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    free(env->map_corners);
    free(env->grid_cells);
    free(env->neighbor_offsets);
    free(env->neighbor_cache_entities);
    free(env->neighbor_cache_indices);
    free(env->static_car_indices);
    free(env->expert_static_car_indices);
    // free(env->map_name);
}

void allocate(Drive *env)
{
    init(env);
    int max_obs = DIM_EGO + DIM_PARTNER * (MAX_CARS - 1) + DIM_ROAD * MAX_ROAD_SEGMENT_OBSERVATIONS;
    env->observations = (float *)calloc(env->active_agent_count * max_obs, sizeof(float));
    env->actions = (int *)calloc(env->active_agent_count, sizeof(int));
    env->rewards = (float *)calloc(env->active_agent_count, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive *env)
{
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

float clip_speed(float speed)
{
    const float max_speed = MAX_SPEED;
    if (speed > max_speed)
        return max_speed;
    if (speed < -max_speed)
        return -max_speed;
    return speed;
}

float wrap_heading(float x)
{
    return atan2f(sinf(x), cosf(x));
}

void move_dynamics(Entity *agent, float acceleration, float steering, int dynamics_model)
{
    if (dynamics_model == CLASSIC)
    {
        // clip acceleration & steering
        float x = agent->x;
        float y = agent->y;
        float heading = agent->heading;
        float vx = agent->vx;
        float vy = agent->vy;

        // Calculate current speed
        float speed = sqrtf(vx * vx + vy * vy);

        // Time step (adjust as needed)
        const float dt = 0.1f;
        // Update speed with acceleration
        speed = speed + 0.5f * acceleration * dt;
        // if (speed < 0) speed = 0;  // Prevent going backward
        speed = clip_speed(speed);
        // compute yaw rate
        float beta = tanhf(.5f * tanf(steering));
        // new heading
        float yaw_rate = (speed * cosf(beta) * tanf(steering)) / agent->length;
        // new velocity
        float new_vx = speed * cosf(heading + beta);
        float new_vy = speed * sinf(heading + beta);
        // Update position
        x = x + (new_vx * dt);
        y = y + (new_vy * dt);
        heading = heading + yaw_rate * dt;
        // heading = normalize_heading(heading);
        // Apply updates to the agent's state
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->vx = new_vx;
        agent->vy = new_vy;
    }
}

float normalize_value(float value, float min, float max)
{
    return (value - min) / (max - min);
}

float reverse_normalize_value(float value, float min, float max)
{
    return value * (max - min) + min;
}

void compute_observations(Drive *env)
{
    int max_obs = DIM_EGO + DIM_PARTNER * (MAX_CARS - 1) + DIM_ROAD * MAX_ROAD_SEGMENT_OBSERVATIONS;
    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    for (int i = 0; i < env->active_agent_count; i++)
    {
        float *obs = &observations[i][0];
        Entity *ego_entity = &env->entities[env->active_agent_indices[i]];
        if (ego_entity->type > 3)
            break;
        // if(ego_entity->respawn_timestep != -1) {
        //     obs[6] = 1;
        //     //continue;
        // }
        float ego_heading = ego_entity->heading;
        float cos_heading = ego_entity->heading_x;
        float sin_heading = ego_entity->heading_y;
        float ego_speed = sqrtf(ego_entity->vx * ego_entity->vx + ego_entity->vy * ego_entity->vy);
        // Set goal distances
        float goal_x = ego_entity->goal_position_x - ego_entity->x;
        float goal_y = ego_entity->goal_position_y - ego_entity->y;
        // Rotate to ego vehicle's frame
        float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
        float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;
        // obs[0] = normalize_value(rel_goal_x, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD);
        // obs[1] = normalize_value(rel_goal_y, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD);
        obs[0] = rel_goal_x * 0.005f;
        obs[1] = rel_goal_y * 0.005f;
        // obs[2] = ego_speed / MAX_SPEED;
        obs[2] = ego_speed * 0.01f;
        obs[3] = ego_entity->width / MAX_VEH_WIDTH;
        obs[4] = ego_entity->length / MAX_VEH_LEN;
        obs[5] = (ego_entity->collision_state > 0) ? 1 : 0;

        // Relative Pos of other cars
        int obs_idx = DIM_EGO; // Start after goal distances
        int cars_seen = 0;
        for (int j = 0; j < MAX_CARS; j++)
        {
            int index = -1;
            if (j < env->active_agent_count)
                index = env->active_agent_indices[j];
            else if (j < env->num_cars)
                index = env->static_car_indices[j - env->active_agent_count];

            if (index == -1)
                continue;
            if (env->entities[index].type > 3)
                break;
            if (index == env->active_agent_indices[i])
                continue; // Skip self, but don't increment obs_idx
            Entity *other_entity = &env->entities[index];
            if (ego_entity->respawn_timestep != -1)
                continue;
            if (other_entity->respawn_timestep != -1)
                continue;
            // Store original relative positions
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            float dist = (dx * dx + dy * dy);
            if (dist > 2500.0f)
                continue;
            // Rotate to ego vehicle's frame
            float rel_x = dx * cos_heading + dy * sin_heading;
            float rel_y = -dx * sin_heading + dy * cos_heading;
            // Store observations with correct indexing
            obs[obs_idx + 0] = rel_x * 0.02f;
            obs[obs_idx + 1] = rel_y * 0.02f;
            obs[obs_idx + 2] = other_entity->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other_entity->length / MAX_VEH_LEN;
            // relative heading
            float rel_heading_x = other_entity->heading_x * ego_entity->heading_x +
                                  other_entity->heading_y * ego_entity->heading_y; // cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
            float rel_heading_y = other_entity->heading_y * ego_entity->heading_x -
                                  other_entity->heading_x * ego_entity->heading_y; // sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
            float rel_heading = atan2f(rel_heading_y, rel_heading_x);
            // obs[obs_idx + 4] = rel_heading;

            obs[obs_idx + 4] = rel_heading_x;
            obs[obs_idx + 5] = rel_heading_y;
            // obs[obs_idx + 4] = cosf(rel_heading) / MAX_ORIENTATION_RAD;
            // obs[obs_idx + 5] = sinf(rel_heading) / MAX_ORIENTATION_RAD;
            // // relative speed
            float other_speed = sqrtf(other_entity->vx * other_entity->vx + other_entity->vy * other_entity->vy);
            obs[obs_idx + 6] = other_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += DIM_PARTNER; // Move to next observation slot
        }
        int remaining_partner_obs = (MAX_CARS - 1 - cars_seen) * DIM_PARTNER;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;
        // map observations
        int entity_list[MAX_ROAD_SEGMENT_OBSERVATIONS * 2]; // Array big enough for all neighboring cells
        int grid_idx = get_grid_index(env, ego_entity->x, ego_entity->y);
        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);
        for (int k = 0; k < list_size; k++)
        {
            int entity_idx = entity_list[k * 2];
            int geometry_idx = entity_list[k * 2 + 1];
            Entity *entity = &env->entities[entity_idx];
            float start_x = entity->traj_x[geometry_idx];
            float start_y = entity->traj_y[geometry_idx];
            float end_x = entity->traj_x[geometry_idx + 1];
            float end_y = entity->traj_y[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego_entity->x;
            float rel_y = mid_y - ego_entity->y;
            float x_obs = rel_x * cos_heading + rel_y * sin_heading;
            float y_obs = -rel_x * sin_heading + rel_y * cos_heading;
            float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);
            float width = 0.1;
            // Calculate angle from ego to midpoint (vector from ego to midpoint)
            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float dx_norm = dx;
            float dy_norm = dy;
            float hypot = sqrtf(dx * dx + dy * dy);
            if (hypot > 0)
            {
                dx_norm /= hypot;
                dy_norm /= hypot;
            }
            // Compute sin and cos of relative angle directly without atan2f
            float cos_angle = dx_norm * cos_heading + dy_norm * sin_heading;
            float sin_angle = -dx_norm * sin_heading + dy_norm * cos_heading;
            obs[obs_idx] = x_obs * 0.02f;
            obs[obs_idx + 1] = y_obs * 0.02f;
            obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 3] = width / MAX_ROAD_SCALE;
            obs[obs_idx + 4] = cos_angle;
            obs[obs_idx + 5] = sin_angle;
            obs[obs_idx + 6] = entity->type - 4.0f;
            obs_idx += DIM_ROAD;
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * DIM_ROAD;
        // Set the entire block to 0 at once
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }
}

void c_reset(Drive *env)
{
    env->timestep = 0;
    set_start_position(env);
    for (int x = 0; x < env->active_agent_count; x++)
    {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].reached_goal = 0;
        env->entities[agent_idx].reached_goal_in_horizon = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].reached_goal_this_episode = 0;
        collision_check(env, agent_idx);
    }
    compute_observations(env);
}

void respawn_agent(Drive *env, int agent_idx)
{
    env->entities[agent_idx].x = env->entities[agent_idx].traj_x[0];
    env->entities[agent_idx].y = env->entities[agent_idx].traj_y[0];
    env->entities[agent_idx].heading = env->entities[agent_idx].traj_heading[0];
    env->entities[agent_idx].heading_x = cosf(env->entities[agent_idx].heading);
    env->entities[agent_idx].heading_y = sinf(env->entities[agent_idx].heading);
    env->entities[agent_idx].vx = env->entities[agent_idx].traj_vx[0];
    env->entities[agent_idx].vy = env->entities[agent_idx].traj_vy[0];
    env->entities[agent_idx].reached_goal = 0;
    env->entities[agent_idx].reached_goal_in_horizon = 0;
    env->entities[agent_idx].respawn_timestep = env->timestep;
}

float compute_control_commands(Entity *agent, float dx, float dy, float *accel, float *steer)
{
    float distance = sqrtf(dx * dx + dy * dy);
    float target_heading = atan2f(dy, dx);
    float heading_error = wrap_heading(target_heading - agent->heading);
    float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    float desired_speed = fminf(distance / 0.1f, MAX_SPEED); // Simple proportional controller for speed
    float speed_error = desired_speed - speed;

    // Proportional control for acceleration and steering
    const float Kp_accel = 1.0f;  // Proportional gain for acceleration
    const float Kp_steer = 2.0f;  // Proportional gain for steering

    *accel = Kp_accel * speed_error;
    *steer = Kp_steer * heading_error;

    // Clip the commands to their limits using fminf and fmaxf for conciseness
    *accel = fmaxf(MIN_ACCELERATION, fminf(MAX_ACCELERATION, *accel));
    *steer = fmaxf(MIN_STEERING_ANGLE, fminf(MAX_STEERING_ANGLE, *steer));
    return distance;
}

void c_step(Drive *env)
{
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    env->timestep++;
    if (DEBUG) printf("timestep: %d\n", env->timestep);
    if (env->timestep == TRAJECTORY_LENGTH)
    {
        add_log(env);
        c_reset(env);
        return;
    }

    for (int i = 0; i < env->active_agent_count; i++)
    {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;
        env->entities[agent_idx].reached_goal_in_horizon = env->entities[agent_idx].reached_goal;
    }

    for (int k = 0; k < TRAJ_VOCAB_LEN; k++)
    {
        float temporal_discount = powf(DISCOUNT_FACTOR, (float) k) / (float) TRAJ_VOCAB_LEN; 
        bool is_immediate_next_step = (k == 0); // when k > 0 the collisions are virtural

        if (DEBUG) printf("\ttemporal_discount: %f\n", temporal_discount);

        for (int i = 0; i < env->expert_static_car_count; i++)
        {
            int expert_idx = env->expert_static_car_indices[i];
            if (env->entities[expert_idx].x == -10000.0f)
                continue;
            if (env->timestep + k < TRAJECTORY_LENGTH)
                move_expert(env, expert_idx, env->timestep + k);
        }

        // Process actions for all active agents
        for (int i = 0; i < env->active_agent_count; i++)
        {
            int agent_idx = env->active_agent_indices[i];
            // Extract action components directly from the multi-discrete action array
            // int (*action_array) = (int (*))env->actions;
            // int traj_index = action_array[i];
            // //printf("traj_index: %d\n", traj_index);
            // // int steer_idx = action_array[i][1];
            // float local_x = T3_GET(&traj_vocab, traj_index, k, 0);
            // float local_y = T3_GET(&traj_vocab, traj_index, k, 1);

            Entity *agent = &env->entities[agent_idx];
            int (*action_array)[2] = (int (*)[2])env->actions;

            int steering_index = action_array[i][0];
            int acceleration_index = action_array[i][1];
            // printf("acceleration_index: %d, steering_index: %d\n", acceleration_index, steering_index);
            float accel = ACCELERATION_VALUES[acceleration_index];
            float steer = STEERING_VALUES[steering_index];
            // float cos_h = agent->heading_x;
            // float sin_h = agent->heading_y;

            // Rotate and translate the local trajectory point to the global frame
            // float dx = local_x * cos_h - local_y * sin_h;
            // float dy = local_x * sin_h + local_y * cos_h;

            // float accel = 0.0f;
            // float steer = 0.0f;

            // compute_control_commands(agent, dx, dy, &accel, &steer);
            // printf("accel: %f, steer: %f\n", accel, steer);
            // printf("x: %f, y: %f\n", agent->x, agent->y);
            move_dynamics(agent, accel, steer, env->dynamics_model);
        }

        for (int i = 0; i < env->active_agent_count; i++)
        {
            int agent_idx = env->active_agent_indices[i];
            int pre_collision_state = env->entities[agent_idx].collision_state;
            
            if (pre_collision_state == NO_COLLISION) { // if no collision before
                collision_check(env, agent_idx); // check the collision
                int collision_state = env->entities[agent_idx].collision_state;

                switch (collision_state)
                {
                case VEHICLE_COLLISION:
                    if (is_immediate_next_step)
                        env->logs[i].collision_rate = 1.0f;

                    if (env->entities[agent_idx].respawn_timestep != -1)
                    {
                        env->rewards[i] += env->reward_vehicle_collision_post_respawn * temporal_discount;
                        env->logs[i].episode_return += env->reward_vehicle_collision_post_respawn * temporal_discount;
                    }
                    else
                    {
                        if (is_immediate_next_step) 
                            env->logs[i].clean_collision_rate = 1.0f;

                        env->rewards[i] += env->reward_vehicle_collision * temporal_discount;
                        env->logs[i].episode_return += env->reward_vehicle_collision * temporal_discount;

                    }
                    if (DEBUG) printf("\tagent_idx: %d, k: %d, reward: %f, collision_state: %d\n", agent_idx, k, env->rewards[i], collision_state);

                    break;
                case OFFROAD:
                    if (is_immediate_next_step) 
                        env->logs[i].offroad_rate = 1.0f;

                    env->rewards[i] += env->reward_offroad_collision * temporal_discount;
                    env->logs[i].episode_return += env->reward_offroad_collision * temporal_discount;
                    if (DEBUG) printf("\tagent_idx: %d, k: %d, reward: %f, collision_state: %d\n", agent_idx, k, env->rewards[i], collision_state);
                    break;
                
                default:
                    break;
                }
                
                if (collision_state > NO_COLLISION && !env->entities[agent_idx].reached_goal_this_episode && is_immediate_next_step)
                    env->entities[agent_idx].collided_before_goal = 1;
            }

            float distance_to_goal = relative_distance_2d(
                env->entities[agent_idx].x,
                env->entities[agent_idx].y,
                env->entities[agent_idx].goal_position_x,
                env->entities[agent_idx].goal_position_y);

            if (env->entities[agent_idx].reached_goal_in_horizon == 0 && (distance_to_goal < RADIUS_GOAL))
            {
                if (env->entities[agent_idx].respawn_timestep != -1)
                {
                    env->rewards[i] += env->reward_goal_post_respawn * temporal_discount;
                    env->logs[i].episode_return += env->reward_goal_post_respawn * temporal_discount;
                }
                else
                {
                    env->rewards[i] += 1.0f * temporal_discount;
                    env->logs[i].episode_return += 1.0f * temporal_discount;
                }
                env->entities[agent_idx].reached_goal_in_horizon = 1;
            }
            
            if (is_immediate_next_step && (distance_to_goal < RADIUS_GOAL))
            {
                env->entities[agent_idx].reached_goal = 1;
                env->entities[agent_idx].reached_goal_this_episode = 1;
            }
        }


        if (is_immediate_next_step)
        {
            for (int i = 0; i < env->active_agent_count; i++)
            {
                int agent_idx = env->active_agent_indices[i];
                int reached_goal = env->entities[agent_idx].reached_goal;
                int collision_state = env->entities[agent_idx].collision_state;
                if (reached_goal)
                {
                    respawn_agent(env, agent_idx);
                    // env->entities[agent_idx].x = -10000;
                    // env->entities[agent_idx].y = -10000;
                    // env->entities[agent_idx].respawn_timestep = env->timestep;
                }
            }
            
            for (int i = 0; i < env->active_agent_count; i++){
                int agent_idx = env->active_agent_indices[i];
                entity_save(&env->entities[agent_idx]);
            }
            for (int i = 0; i < env->expert_static_car_count; i++)
            {
                int expert_idx = env->expert_static_car_indices[i];
                entity_save(&env->entities[expert_idx]);
            }
        }
    }

    // Recover the stored state as next state

    for (int i = 0; i < env->active_agent_count; i++){
        int agent_idx = env->active_agent_indices[i];
        entity_restore(&env->entities[agent_idx]);
    }
    for (int i = 0; i < env->expert_static_car_count; i++)
    {
        int expert_idx = env->expert_static_car_indices[i];
        entity_restore(&env->entities[expert_idx]);
    }

    compute_observations(env);
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct Client Client;
struct Client
{
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    int car_assignments[MAX_CARS]; // To keep car model assignments consistent per vehicle
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client *make_client(Drive *env)
{
    Client *client = (Client *)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferLib Ray GPU Drive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    for (int i = 0; i < MAX_CARS; i++)
    {
        client->car_assignments[i] = (rand() % 4) + 1;
    }
    // Get initial target position from first active agent
    float map_center_x = (env->map_corners[0] + env->map_corners[2]) / 2.0f;
    float map_center_y = (env->map_corners[1] + env->map_corners[3]) / 2.0f;
    Vector3 target_pos = {
        0,
        0, // Y is up
        1  // Z is depth
    };

    // Set up camera to look at target from above and behind
    client->default_camera_position = (Vector3){
        0,      // Same X as target
        120.0f, // 20 units above target
        175.0f  // 20 units behind target
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f}; // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

// Camera control functions
void handle_camera_controls(Client *client)
{
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
    {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
    {
        is_dragging = false;
    }

    if (is_dragging)
    {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {
            (current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed};

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;

        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0)
    {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z};

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void draw_agent_obs(Drive *env, int agent_index)
{
    // Diamond dimensions
    float diamond_height = 3.0f; // Total height of diamond
    float diamond_width = 1.5f;  // Width of diamond
    float diamond_z = 8.0f;      // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height / 2};    // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height / 2}; // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width / 2, diamond_z};          // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width / 2, diamond_z};          // Back point
    Vector3 left_point = (Vector3){-diamond_width / 2, 0.0f, diamond_z};          // Left point
    Vector3 right_point = (Vector3){diamond_width / 2, 0.0f, diamond_z};          // Right point

    // Draw the diamond faces
    // Top pyramid
    DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN); // Front-right face
    DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);  // Back-right face
    DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);   // Back-left face
    DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);  // Front-left face

    // Bottom pyramid
    DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
    DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
    DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
    DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    if (!IsKeyDown(KEY_LEFT_CONTROL))
    {
        return;
    }
    int max_obs = DIM_EGO + DIM_PARTNER * (MAX_CARS - 1) + DIM_PARTNER * MAX_ROAD_SEGMENT_OBSERVATIONS;
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    float *agent_obs = &observations[agent_index][0];
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;
    DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, GREEN);
    // First draw other agent observations
    int obs_idx = DIM_EGO; // Start after goal distances
    for (int j = 0; j < MAX_CARS - 1; j++)
    {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0)
        {
            obs_idx += DIM_PARTNER; // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        DrawLine3D(
            (Vector3){0, 0, 0},
            (Vector3){x, y, 1},
            ORANGE);
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 7.5f;
        float arrow_x = x + arrow_length * cosf(partner_angle);
        float arrow_y = y + arrow_length * sinf(partner_angle);
        DrawLine3D((Vector3){x, y, 1}, (Vector3){arrow_x, arrow_y, 1}, PUFF_WHITE);
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 2.0f; // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx * dx + dy * dy);
        if (length > 0)
        {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector
            float px = -dy * arrow_size;
            float py = dx * arrow_size;

            // Draw the two lines forming the arrow head
            DrawLine3D(
                (Vector3){arrow_x, arrow_y, 1},
                (Vector3){arrow_x - dx * arrow_size + px, arrow_y - dy * arrow_size + py, 1},
                PUFF_WHITE);
            DrawLine3D(
                (Vector3){arrow_x, arrow_y, 1},
                (Vector3){arrow_x - dx * arrow_size - px, arrow_y - dy * arrow_size - py, 1},
                PUFF_WHITE);
        }
        obs_idx += 7; // Move to next agent observation (7 values per agent)
    }
    // Then draw map observations
    int map_start_idx = DIM_EGO + DIM_PARTNER * (MAX_CARS - 1); // Start after agent observations
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++)
    { // Loop through potential map entities
        int entity_idx = map_start_idx + k * DIM_ROAD;
        if (agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0)
        {
            continue;
        }
        Color lineColor = BLUE; // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        if (entity_type + 4 != ROAD_EDGE)
        {
            continue;
        }
        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 4]);
        float rel_angle_y = (agent_obs[entity_idx + 5]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);
        DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
        DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y)
{
    Color CURB_TOP = (Color){220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    // Calculate curb dimensions
    float curb_height = 0.5f; // Height of the curb
    float curb_width = 0.3f;  // Width/thickness of the curb

    // Calculate direction vector between start and end
    Vector3 direction = {
        end_x - start_x,
        end_y - start_y,
        0.0f};

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {
        direction.x / length,
        direction.y / length,
        0.0f};

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {
        -normalized_dir.y,
        normalized_dir.x,
        0.0f};

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {
        start_x - perpendicular.x * curb_width / 2,
        start_y - perpendicular.y * curb_width / 2,
        1.0f};
    Vector3 b2 = {
        start_x + perpendicular.x * curb_width / 2,
        start_y + perpendicular.y * curb_width / 2,
        1.0f};
    Vector3 b3 = {
        end_x + perpendicular.x * curb_width / 2,
        end_y + perpendicular.y * curb_width / 2,
        1.0f};
    Vector3 b4 = {
        end_x - perpendicular.x * curb_width / 2,
        end_y - perpendicular.y * curb_width / 2,
        1.0f};

    // Draw the curb faces
    // Bottom face
    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    // Top face (raised by curb_height)
    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    // Side faces
    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void c_render(Drive *env)
{
    if (env->client == NULL)
    {
        env->client = make_client(env);
    }
    Client *client = env->client;
    BeginDrawing();
    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(client->camera);
    handle_camera_controls(env->client);

    // Draw a grid to help with orientation
    // DrawGrid(20, 1.0f);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[1], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[0], env->map_corners[3], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[2], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[3], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, PUFF_CYAN);
    for (int i = 0; i < env->num_entities; i++)
    {
        // Draw cars
        if (env->entities[i].type == 1 || env->entities[i].type == 2)
        {
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            bool is_static_car = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++)
            {
                if (env->active_agent_indices[j] == i)
                {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for (int j = 0; j < env->static_car_count; j++)
            {
                if (env->static_car_indices[j] == i)
                {
                    is_static_car = true;
                    break;
                }
            }
            // HIDE CARS ON RESPAWN - IMPORTANT TO KNOW VISUAL SETTING
            if (!is_active_agent && !is_static_car || env->entities[i].respawn_timestep != -1)
            {
                continue;
            }
            Vector3 position;
            float heading;
            position = (Vector3){
                env->entities[i].x,
                env->entities[i].y,
                1};
            heading = env->entities[i].heading;
            // Create size vector
            Vector3 size = {
                env->entities[i].length,
                env->entities[i].width,
                env->entities[i].height};
            // Save current transform
            rlPushMatrix();
            // Translate to position, rotate around Y axis, then draw
            rlTranslatef(position.x, position.y, position.z);
            rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f); // Convert radians to degrees
            // Determine color based on active status and other conditions
            Color object_color = PUFF_BACKGROUND2; // Default color for non-active vehicles
            Color outline_color = PUFF_CYAN;
            Model car_model = client->cars[5];
            if (is_active_agent)
            {
                car_model = client->cars[client->car_assignments[i % 64]];
            }
            if (agent_index == env->human_agent_idx)
            {
                object_color = PUFF_CYAN;
                outline_color = PUFF_WHITE;
            }
            if (is_active_agent && env->entities[i].collision_state > 0)
            {
                car_model = client->cars[0]; // Collided agent
            }
            // Draw obs for human selected agent
            if (agent_index == env->human_agent_idx && !env->entities[agent_index].reached_goal)
            {
                draw_agent_obs(env, agent_index);
            }
            // Draw cube for cars static and active
            // Calculate scale factors based on desired size and model dimensions

            BoundingBox bounds = GetModelBoundingBox(car_model);
            Vector3 model_size = {
                bounds.max.x - bounds.min.x,
                bounds.max.y - bounds.min.y,
                bounds.max.z - bounds.min.z};
            Vector3 scale = {
                size.x / model_size.x,
                size.y / model_size.y,
                size.z / model_size.z};
            DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
            rlPopMatrix();

            float cos_heading = env->entities[i].heading_x;
            float sin_heading = env->entities[i].heading_y;

            // Calculate half dimensions
            float half_len = env->entities[i].length * 0.5f;
            float half_width = env->entities[i].width * 0.5f;

            // Calculate the four corners of the collision box
            Vector3 corners[4] = {
                (Vector3){
                    position.x + (half_len * cos_heading - half_width * sin_heading),
                    position.y + (half_len * sin_heading + half_width * cos_heading),
                    position.z},
                (Vector3){
                    position.x + (half_len * cos_heading + half_width * sin_heading),
                    position.y + (half_len * sin_heading - half_width * cos_heading),
                    position.z},
                (Vector3){
                    position.x + (-half_len * cos_heading - half_width * sin_heading),
                    position.y + (-half_len * sin_heading + half_width * cos_heading),
                    position.z},
                (Vector3){
                    position.x + (-half_len * cos_heading + half_width * sin_heading),
                    position.y + (-half_len * sin_heading - half_width * cos_heading),
                    position.z}};

            // Draw the corners as spheres
            /*
            for(int j = 0; j < 4; j++) {
                DrawSphere(corners[j], 0.3f, RED);  // Draw red spheres at each corner
            }
            */
            for (int j = 0; j < 4; j++)
            {
                DrawLine3D(corners[j], corners[(j + 1) % 4], PURPLE); // Draw red lines between corners
            }
            // FPV Camera Control
            if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index)
            {
                if (env->entities[agent_index].reached_goal)
                {
                    env->human_agent_idx = rand() % env->active_agent_count;
                }
                Vector3 camera_position = (Vector3){
                    position.x - (25.0f * cosf(heading)),
                    position.y - (25.0f * sinf(heading)),
                    position.z + 15};

                Vector3 camera_target = (Vector3){
                    position.x + 40.0f * cosf(heading),
                    position.y + 40.0f * sinf(heading),
                    position.z - 5.0f};
                client->camera.position = camera_position;
                client->camera.target = camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            if (IsKeyReleased(KEY_SPACE))
            {
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            // Draw goal position for active agents

            if (!is_active_agent || env->entities[i].valid == 0)
            {
                continue;
            }
            if (!IsKeyDown(KEY_LEFT_CONTROL))
            {
                DrawSphere((Vector3){
                               env->entities[i].goal_position_x,
                               env->entities[i].goal_position_y,
                               1},
                           0.5f, DARKGREEN);
            }
        }
        // Draw road elements
        if (env->entities[i].type <= 3 && env->entities[i].type >= 7)
        {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++)
        {
            Vector3 start = {
                env->entities[i].traj_x[j],
                env->entities[i].traj_y[j],
                1};
            Vector3 end = {
                env->entities[i].traj_x[j + 1],
                env->entities[i].traj_y[j + 1],
                1};
            Color lineColor = GRAY;
            if (env->entities[i].type == ROAD_LANE)
                lineColor = GRAY;
            else if (env->entities[i].type == ROAD_LINE)
                lineColor = BLUE;
            else if (env->entities[i].type == ROAD_EDGE)
                lineColor = WHITE;
            else if (env->entities[i].type == DRIVEWAY)
                lineColor = RED;
            if (env->entities[i].type != ROAD_EDGE)
            {
                continue;
            }
            if (!IsKeyDown(KEY_LEFT_CONTROL))
            {
                draw_road_edge(env, start.x, start.y, end.x, end.y);
                // DrawLine3D(start, end, lineColor);
                // DrawCube(start, 0.5f, 0.5f, 0.5f, lineColor);
                // DrawCube(end, 0.5f, 0.5f, 0.5f, lineColor);
            }
        }
    }
    // Draw grid cells using the stored bounds
    float grid_start_x = env->map_corners[0];
    float grid_start_y = env->map_corners[1];
    for (int i = 0; i < env->grid_cols; i++)
    {
        for (int j = 0; j < env->grid_rows; j++)
        {
            float x = grid_start_x + i * GRID_CELL_SIZE;
            float y = grid_start_y + j * GRID_CELL_SIZE;
            // int index = i * env->grid_rows + j;
            DrawCubeWires(
                (Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 1},
                GRID_CELL_SIZE, GRID_CELL_SIZE, 0.1f, PUFF_BACKGROUND2);
        }
    }
    EndMode3D();
    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)",
                        client->camera.position.x,
                        client->camera.position.y,
                        client->camera.position.z),
             10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)",
                        client->camera.target.x,
                        client->camera.target.y,
                        client->camera.target.z),
             10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);
    // acceleration & steering
    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);
    // Controls help
    DrawText("Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent",
             10, client->height - 30, 20, PUFF_WHITE);
    // acceleration & steering
    DrawText(TextFormat("Acceleration: %d", env->actions[env->human_agent_idx * 2]), 10, 110, 20, PUFF_WHITE);
    DrawText(TextFormat("Steering: %d", env->actions[env->human_agent_idx * 2 + 1]), 10, 130, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Rows: %d", env->grid_rows), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_cols), 10, 170, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client *client)
{
    for (int i = 0; i < 6; i++)
    {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}