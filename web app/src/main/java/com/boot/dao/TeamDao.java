package com.boot.dao;

import java.util.List;

import org.apache.ibatis.annotations.Param;

import com.boot.entity.Team;
import com.boot.entity.User;

public interface TeamDao {

	List<Team> getList(@Param("userId")Integer userId);

	Team getDeatil(@Param("teamId")Integer teamId);

	List<User> getTeamUser(@Param("teamId")Integer teamId);

	Integer countByTeamUser(@Param("teamId")Integer teamId,@Param("userId") Integer userId);

	void insertTeamUser(@Param("teamId")Integer teamId,@Param("userId") Integer userId);

	Team getTeamDefault(@Param("userId")Integer userId);

	void removeUser(@Param("teamId")Integer teamId,@Param("userId") Integer userId);

}
